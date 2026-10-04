"""OpenRainER (Emilia-Romagna, 2021-2022) reduced to the `common` format.

The archive holds monthly CML files (151 links x 2 sublinks, `tsl` and `rsl`
at 1 min, UTC) and monthly weather products: the AWS gauge network (15-min
rain depth) and the radar rain maps `RADrain` and `RADadj` (15-min rain
depth on a ~1 km grid). `RADadj` is the radar adjusted to the gauges by
kriging. It is the reference used here, averaged over the grid pixels the
link path crosses.

Months whose `RADadj` file is not extracted under `weather/rad_adj/` are read
from `_download/RADadj.tar` through a temporary file, which is deleted after
use. Nothing under the archive is modified.

The radar and AWS time stamps mark the end of the 15-min accumulation. The
reference written here is labelled by the bin start, so a radar value stamped
12:15 becomes the bin [12:00, 12:15). A lag scan of attenuation against
radar rain peaks at this alignment.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from physprior.cml.sources import common
from physprior.units import require

NAME = "openrainer"
N_MONTHS = 6
MAX_SUBLINKS = 40
FREQ_GHZ = (15.0, 40.0)
LENGTH_KM = (0.3, 20.0)
MIN_AVAILABILITY = 0.8
GAUGE_RADIUS_KM = 3.0
REF_MINUTES = 15
# Received levels at or below this are the receiver floor (outage), not a
# measurement. The archive floor sits at -100 and -101 dBm.
RSL_FLOOR_DBM = -99.0
SPIKE_DB = 40.0
SENTINELS = (-99.9, 255.0, -9999.0)
PATH_STEP_KM = 0.25


def archive() -> Path:
    return common.data_root() / NAME


def drop_spikes(a: np.ndarray, jump: float = SPIKE_DB) -> np.ndarray:
    """Mask of single samples that jump by more than `jump` dB and revert.

    Works along the last axis. A sample is flagged when it differs from both
    neighbours by more than `jump` with opposite signs, the signature of a
    corrupted reading rather than of rain, which builds up over minutes.
    """
    mask = np.zeros(a.shape, dtype=bool)
    d1 = a[..., 1:-1] - a[..., :-2]
    d2 = a[..., 2:] - a[..., 1:-1]
    with np.errstate(invalid="ignore"):
        spike = (np.abs(d1) > jump) & (np.abs(d2) > jump) & (np.sign(d1) != np.sign(d2))
    mask[..., 1:-1] = spike
    return mask


def quant_step(values: np.ndarray, n_top: int = 50) -> float:
    """Quantisation step of a level series, in dB.

    The smallest gap between the `n_top` most frequent distinct values,
    rounded to 0.1 dB. Using the most frequent values ignores the rare
    in-between levels that appear when a logger averages several raw
    readings into one sample.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    vals, counts = np.unique(np.round(v, 4), return_counts=True)
    top = np.sort(vals[np.argsort(-counts)[:n_top]])
    gaps = np.diff(top)
    gaps = gaps[gaps > 1e-6]
    return float(np.round(gaps.min(), 1)) if gaps.size else 0.0


def clean_levels(
    tsl: np.ndarray | None,
    rsl: np.ndarray,
    floor: float = RSL_FLOOR_DBM,
    ceiling: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Attenuation (tsl - rsl, or -rsl) with impossible values set to NaN.

    Flags, in order: sentinel values, rsl at or below the receiver floor,
    rsl at or above `ceiling` (when given), rsl at or above tsl, and
    single-sample spikes of more than 40 dB.
    Returns the attenuation and the number of samples each flag removed.
    """
    rsl = rsl.astype(float, copy=True)
    counts: dict[str, int] = {}
    bad = np.zeros(rsl.shape, dtype=bool)
    for s in SENTINELS:
        bad |= np.isclose(rsl, s)
        if tsl is not None:
            bad |= np.isclose(tsl, s)
    counts["sentinel"] = int(bad.sum())
    with np.errstate(invalid="ignore"):
        floor_bad = rsl <= floor
    counts["floor"] = int((floor_bad & ~bad).sum())
    bad |= floor_bad
    if ceiling is not None:
        with np.errstate(invalid="ignore"):
            high = rsl >= ceiling
        counts["ceiling"] = int((high & ~bad).sum())
        bad |= high
    if tsl is not None:
        with np.errstate(invalid="ignore"):
            above = rsl >= tsl
        counts["rsl_ge_tsl"] = int((above & ~bad).sum())
        bad |= above
        attn = tsl - rsl
    else:
        attn = -rsl
    attn[bad] = np.nan
    spikes = drop_spikes(attn)
    counts["spike"] = int(spikes.sum())
    attn[spikes] = np.nan
    return attn, counts


def path_pixels(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
    length_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the grid pixels a straight path crosses.

    The path is sampled every 0.25 km and each point is assigned to the
    nearest pixel centre; duplicates are removed.
    """
    n = max(2, int(np.ceil(length_km / PATH_STEP_KM)) + 1)
    f = np.linspace(0.0, 1.0, n)
    plat = lat0 + f * (lat1 - lat0)
    plon = lon0 + f * (lon1 - lon0)
    iy = np.abs(lat[:, None] - plat[None, :]).argmin(0)
    ix = np.abs(lon[:, None] - plon[None, :]).argmin(0)
    px = np.unique(np.stack([iy, ix], axis=1), axis=0)
    return px[:, 0], px[:, 1]


def _cml_files(root: Path) -> dict[str, Path]:
    return {p.name[4:10]: p for p in sorted((root / "cml").glob("CML_*.nc"))}


def _link_table(ds: xr.Dataset) -> pd.DataFrame:
    """One row per sublink, units converted to GHz and km."""
    rows = []
    for c in ds["cml_id"].values:
        one = ds.sel(cml_id=c)
        for s in ds["sublink_id"].values:
            sub = one.sel(sublink_id=s)
            pol = str(sub["polarization"].values).strip().lower()
            rows.append(
                {
                    "cml_id": str(c),
                    "sublink_id": str(s),
                    "link_id": f"{c}_{s}",
                    "freq_ghz": float(sub["frequency"]) / 1000.0,
                    "pol": {"horizontal": "H", "vertical": "V"}.get(pol, ""),
                    "length_km": float(one["length"]) / 1000.0,
                    "lat_a": float(one["site_0_lat"]),
                    "lon_a": float(one["site_0_lon"]),
                    "lat_b": float(one["site_1_lat"]),
                    "lon_b": float(one["site_1_lon"]),
                }
            )
    return pd.DataFrame(rows)


def _aws_month_rain(root: Path, links: pd.DataFrame) -> pd.Series:
    """Mean monthly rain depth (mm) at the AWS gauges near candidate links."""
    mid_lat = ((links["lat_a"] + links["lat_b"]) / 2).to_numpy()
    mid_lon = ((links["lon_a"] + links["lon_b"]) / 2).to_numpy()
    out = {}
    for p in sorted((root / "weather" / "aws").glob("AWS_*.nc")):
        with xr.open_dataset(p) as ds:
            d = common.haversine_km(
                ds["latitude"].values[:, None],
                ds["longitude"].values[:, None],
                mid_lat[None, :],
                mid_lon[None, :],
            )
            near = d.min(axis=1) <= GAUGE_RADIUS_KM
            r = ds["rainfall_amount"].values[near]
            r = np.where(r >= 0, r, np.nan)
            out[p.stem[-6:]] = float(np.nanmean(np.nansum(r, axis=1)))
    return pd.Series(out).sort_index()


@contextmanager
def _radar_file(root: Path, ym: str) -> Iterator[Path]:
    """Path to the RADadj file of month `ym`, unpacking it from the tar if needed."""
    p = root / "weather" / "rad_adj" / f"RADadj_{ym}.nc"
    if p.exists():
        yield p
        return
    tar_path = root / "_download" / "RADadj.tar"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / f"RADadj_{ym}.nc"
        with tarfile.open(tar_path) as tar:
            member = tar.extractfile(f"RADadj_{ym}.nc.gz")
            if member is None:
                raise FileNotFoundError(f"RADadj_{ym}.nc.gz not in {tar_path}")
            with gzip.GzipFile(fileobj=member) as gz, dst.open("wb") as fh:
                shutil.copyfileobj(gz, fh, length=1 << 24)
        yield dst


def _radar_reference(
    path: Path, links: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[str, float]]]:
    """Path-averaged RADadj rain rate (mm/h) per link, labelled by bin start."""
    with netCDF4.Dataset(path) as nc:
        lat = np.asarray(nc["lat"][:], dtype=float)
        lon = np.asarray(nc["lon"][:], dtype=float)
        t = np.asarray(nc["time"][:], dtype="int64")
        units = nc["time"].units
        require_units = units.startswith("seconds since 1970-01-01")
        require(require_units, f"unexpected radar time units {units!r}")
        lats = np.r_[links["lat_a"], links["lat_b"]]
        lons = np.r_[links["lon_a"], links["lon_b"]]
        y0 = max(int(np.searchsorted(lat, lats.min())) - 3, 0)
        y1 = int(np.searchsorted(lat, lats.max())) + 3
        x0 = max(int(np.searchsorted(lon, lons.min())) - 3, 0)
        x1 = int(np.searchsorted(lon, lons.max())) + 3
        var = nc["rainfall_amount"]
        var.set_auto_mask(True)
        box = np.ma.filled(var[:, y0:y1, x0:x1].astype("float32"), np.nan)
    box[box < 0] = np.nan
    blat, blon = lat[y0:y1], lon[x0:x1]
    # Stamps mark the end of the 15-min accumulation: label by bin start.
    start = (pd.to_datetime(t, unit="s") - pd.Timedelta(minutes=REF_MINUTES)).astype(
        "datetime64[ns]"
    )
    frames = []
    info: dict[str, tuple[str, float]] = {}
    for row in links.itertuples():
        iy, ix = path_pixels(
            blat, blon, row.lat_a, row.lon_a, row.lat_b, row.lon_b, row.length_km
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN time steps
            depth = np.nanmean(box[:, iy, ix], axis=1)
        rate = depth * (60.0 / REF_MINUTES)  # mm per 15 min -> mm/h
        frames.append(
            pd.DataFrame({"link_id": row.link_id, "time": start, "rain_mmh": rate})
        )
        mid_lat = (row.lat_a + row.lat_b) / 2
        mid_lon = (row.lon_a + row.lon_b) / 2
        dist = common.haversine_km(mid_lat, mid_lon, blat[iy], blon[ix])
        ids = ";".join(
            f"rad_adj:{y0 + a}_{x0 + b}" for a, b in zip(iy, ix, strict=True)
        )
        info[row.link_id] = (ids, float(np.mean(dist)))
    return pd.concat(frames, ignore_index=True), info


def build(out: Path | None = None) -> Path:
    """Read the OpenRainER archive and write the four `common` files."""
    root = archive()
    out = out or common.out_dir(NAME)
    files = _cml_files(root)
    require(bool(files), f"no CML files under {root / 'cml'}")

    def candidates(path: Path) -> pd.DataFrame:
        with xr.open_dataset(path) as ds:
            links = _link_table(ds)
        return links[
            links["freq_ghz"].between(*FREQ_GHZ)
            & links["length_km"].between(*LENGTH_KM)
            & links["pol"].isin(["H", "V"])
        ].copy()

    month_rain = _aws_month_rain(root, candidates(next(iter(files.values()))))
    months = sorted(month_rain.sort_values(ascending=False).index[:N_MONTHS])
    cand = candidates(files[months[0]])

    # Clean attenuation for every candidate sublink over the chosen months.
    attn_parts: dict[str, list[pd.Series]] = {k: [] for k in cand["link_id"]}
    freq_changed: set[str] = set()
    flags: dict[str, int] = {}
    level_sample: list[np.ndarray] = []
    n_minutes = 0
    sources: list[str] = []
    for ym in months:
        p = files[ym]
        sources.append(str(p.relative_to(root)))
        with xr.open_dataset(p) as ds:
            ds = ds.sel(cml_id=sorted(cand["cml_id"].unique()))
            times = pd.DatetimeIndex(ds["time"].values)
            n_minutes += times.size
            rsl = ds["rsl"].transpose("cml_id", "sublink_id", "time").values
            tsl = ds["tsl"].transpose("cml_id", "sublink_id", "time").values
            freq = ds["frequency"].transpose("cml_id", "sublink_id").values
            attn, cnt = clean_levels(tsl, rsl)
            for k, v in cnt.items():
                flags[k] = flags.get(k, 0) + v
            level_sample.append(rsl[:, :, ::97].ravel())
            cml_ids = [str(c) for c in ds["cml_id"].values]
            sub_ids = [str(s) for s in ds["sublink_id"].values]
        for i, c in enumerate(cml_ids):
            for j, s in enumerate(sub_ids):
                lid = f"{c}_{s}"
                if lid not in attn_parts:
                    continue
                f0 = cand.loc[cand["link_id"] == lid, "freq_ghz"].iloc[0]
                if abs(freq[i, j] / 1000.0 - f0) >= 0.01:
                    freq_changed.add(lid)
                attn_parts[lid].append(pd.Series(attn[i, j], index=times))

    avail = {
        k: float(sum(int(s.notna().sum()) for s in v)) / n_minutes
        for k, v in attn_parts.items()
    }
    cand["availability"] = cand["link_id"].map(avail)
    # A sublink retuned to another channel during the period is a different
    # radio before and after; it is left out.
    cand = cand[
        (cand["availability"] >= MIN_AVAILABILITY) & ~cand["link_id"].isin(freq_changed)
    ]
    # One sublink per link (the better available one) for spatial spread,
    # then the best-available links up to the cap.
    cand = cand.sort_values(["availability", "link_id"], ascending=[False, True])
    chosen = cand.drop_duplicates("cml_id").head(MAX_SUBLINKS).copy()
    chosen = chosen.sort_values("link_id").reset_index(drop=True)

    series = pd.concat(
        [
            pd.DataFrame(
                {"link_id": lid, "time": s.index, "attn_db": s.to_numpy()}
            ).dropna()
            for lid in chosen["link_id"]
            for s in attn_parts[lid]
        ],
        ignore_index=True,
    )
    del attn_parts

    ref_frames = []
    info: dict[str, tuple[str, float]] = {}
    for ym in months:
        with _radar_file(root, ym) as rp:
            ref, info = _radar_reference(rp, chosen)
            if rp.is_relative_to(root):
                sources.append(str(rp.relative_to(root)))
            else:
                sources.append(f"_download/RADadj.tar:RADadj_{ym}.nc.gz")
        ref_frames.append(ref)
    reference = pd.concat(ref_frames, ignore_index=True).dropna()
    reference = reference.sort_values(["link_id", "time"]).reset_index(drop=True)

    quant = quant_step(np.concatenate(level_sample))
    chosen["ref_kind"] = "radar"
    chosen["ref_ids"] = chosen["link_id"].map(lambda k: info[k][0])
    chosen["ref_dist_km"] = chosen["link_id"].map(lambda k: info[k][1])
    chosen["attn_kind"] = "tsl-rsl"
    chosen["quant_db"] = quant

    meta = {
        "dataset": NAME,
        "ref_minutes": REF_MINUTES,
        "series_minutes": 1,
        "start": str(series["time"].min()),
        "end": str(series["time"].max()),
        "n_links": len(chosen),
        "months": months,
        "selection": (
            f"The {N_MONTHS} calendar months with the largest mean rain depth at "
            f"the AWS gauges within {GAUGE_RADIUS_KM:g} km of a candidate link "
            f"midpoint. Sublinks with {FREQ_GHZ[0]:g}-{FREQ_GHZ[1]:g} GHz, "
            f"path length {LENGTH_KM[0]:g}-{LENGTH_KM[1]:g} km and at least "
            f"{MIN_AVAILABILITY:.0%} valid minutes over those months, and the "
            "same frequency in every month. One "
            "sublink per link, the one with more valid minutes, and the "
            f"{MAX_SUBLINKS} best-available links."
        ),
        "sources": sources,
        "month_rain_mm": {k: round(v, 1) for k, v in month_rain.items()},
        "flagged_samples": flags,
        "notes": (
            "attn_db is tsl - rsl in dB at 1 min, time stamps as in the archive "
            "(UTC). Minute values are averages of 1-dB raw readings, so "
            f"fractional levels occur; quant_db ({quant:g} dB) is the step "
            "between the most frequent levels. Set to NaN and dropped: "
            f"rsl <= {RSL_FLOOR_DBM:g} dBm (receiver floor), rsl >= tsl, the "
            f"sentinels {list(SENTINELS)} and single-sample jumps of more than "
            f"{SPIKE_DB:g} dB that revert. rain_mmh is the gauge-adjusted "
            "radar (RADadj, 15-min depth x 4) averaged over the pixels the "
            "path crosses (path sampled every 0.25 km, nearest ~1 km pixel). "
            "Radar stamps mark the end of the accumulation; the reference is "
            "labelled by bin start (stamp - 15 min). ref_dist_km is the mean "
            "distance from the link midpoint to the centres of those pixels. "
            "The six months are not contiguous: start and end bound them and "
            "`months` lists them."
        ),
    }
    common.write(out, chosen, series, reference, meta)
    return out


if __name__ == "__main__":
    print(build())
