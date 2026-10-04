"""Netherlands CML (Overeem et al. 2024, May-August 2012) in the `common` format.

The archive under `<root>/netherlands` holds `monthly/cml_2012-MM.nc`, a
reshaping of the raw RAINLINK file in `_download/IDRawCMLdata.zip`: for each
link and sublink the minimum and maximum received level (dBm) over 15-min
intervals, stamped at the interval end (UTC). There is no transmitted
level. Frequency is stored in GHz and length in km. Polarisation is not
given; the data description says most links are vertical, so `pol` is "V"
for all.

The only reference on disk is the KNMI automatic-station hourly rain sum
(`knmi_hourly/RH_2012-MM.txt`), for June to August. It is a point gauge, so
links are kept only when a station with data lies within 3 km of the link
midpoint. May has CML data but no gauge file, so May has no reference rows.

`series.parquet` keeps the native 15-min sampling, labelled by interval
start (stamp - 15 min) so that it shares the reference's bin-start
convention. `attn_db` is -(Pmin + Pmax) / 2, the negative mid-range level of
the interval. Its absolute value is arbitrary; only changes carry meaning.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from physprior.cml.sources import common
from physprior.cml.sources.openrainer import clean_levels, quant_step
from physprior.units import require

NAME = "netherlands"
MONTHS = ("2012-05", "2012-06", "2012-07", "2012-08")
MAX_SUBLINKS = 40
FREQ_GHZ = (15.0, 40.0)
LENGTH_KM = (0.3, 20.0)
MIN_AVAILABILITY = 0.8
GAUGE_RADIUS_KM = 3.0
SERIES_MINUTES = 15
REF_MINUTES = 60
# Levels at or below -99.9 dBm are the receiver floor or the NEC sentinel
# -99.9; levels at or above -10 dBm are the sentinels 0 and -1.
RSL_FLOOR_DBM = -99.9
RSL_CEILING_DBM = -10.0


def archive() -> Path:
    return common.data_root() / NAME


def read_knmi_hourly(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stations and hourly rain rates from one KNMI `RH` export.

    `HH` is the hour ending at HH:00 UT, so the bin start is HH - 1 h. `RH`
    is the hourly sum in 0.1 mm, with -1 for less than 0.05 mm (set to 0).
    Returns (stations: stn, lon, lat, name) and (stn, time, rain_mmh).
    """
    stations = []
    pat = re.compile(r"#\s+(\d{3})\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(.*)")
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            m = pat.match(line)
            if m:
                stations.append((int(m[1]), float(m[2]), float(m[3]), m[5].strip()))
    st = pd.DataFrame(stations, columns=["stn", "lon", "lat", "name"])
    raw = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=["stn", "date", "hh", "rh"],
        skipinitialspace=True,
        dtype={"stn": int, "date": str, "hh": int},
    )
    rh = pd.to_numeric(raw["rh"], errors="coerce")
    rh = rh.where(rh != -1, 0.0)
    rh = rh.where(rh >= 0)
    start = pd.to_datetime(raw["date"], format="%Y%m%d") + pd.to_timedelta(
        raw["hh"] - 1, unit="h"
    )
    rain = pd.DataFrame(
        {
            "stn": raw["stn"],
            "time": start.astype("datetime64[ns]"),
            "rain_mmh": rh * 0.1,  # 0.1 mm per hour -> mm/h
        }
    ).dropna()
    return st, rain


def _monthly_files(root: Path) -> list[Path]:
    files = [root / "monthly" / f"cml_{m}.nc" for m in MONTHS]
    for f in files:
        require(f.exists(), f"missing {f}")
    return files


def _link_table(ds: xr.Dataset) -> pd.DataFrame:
    require(ds["frequency"].attrs.get("units") == "GHz", "frequency not in GHz")
    require(ds["length"].attrs.get("units") == "km", "length not in km")
    sub = [str(s) for s in ds["sublink_id"].values]
    freq = ds["frequency"].transpose("cml_id", "sublink_id").values
    direction = ds["direction"].transpose("cml_id", "sublink_id").values
    rows = []
    for i, c in enumerate(ds["cml_id"].values):
        for j, s in enumerate(sub):
            if direction[i, j] < 0:
                continue
            rows.append(
                {
                    "cml_id": str(c),
                    "link_id": f"{c}_{s}",
                    "freq_ghz": float(freq[i, j]),
                    "length_km": float(ds["length"].values[i]),
                    "lat_a": float(ds["site_0_lat"].values[i]),
                    "lon_a": float(ds["site_0_lon"].values[i]),
                    "lat_b": float(ds["site_1_lat"].values[i]),
                    "lon_b": float(ds["site_1_lon"].values[i]),
                    "vendor": str(ds["vendor"].values[i]),
                }
            )
    return pd.DataFrame(rows)


def build(out: Path | None = None) -> Path:
    """Read the Netherlands archive and write the four `common` files."""
    root = archive()
    out = out or common.out_dir(NAME)
    files = _monthly_files(root)

    gauge_files = sorted((root / "knmi_hourly").glob("RH_*.txt"))
    require(bool(gauge_files), f"no KNMI hourly files under {root}")
    st_parts, rain_parts = [], []
    for g in gauge_files:
        st, rain = read_knmi_hourly(g)
        st_parts.append(st)
        rain_parts.append(rain)
    stations = pd.concat(st_parts).drop_duplicates("stn")
    rain = pd.concat(rain_parts, ignore_index=True).drop_duplicates(["stn", "time"])
    stations = stations[stations["stn"].isin(rain["stn"].unique())]

    # Candidates: present in the first month, the right band and length, and
    # a gauge with data within the radius of the midpoint.
    with xr.open_dataset(files[0]) as ds0:
        links = _link_table(ds0)
    mid_lat = ((links["lat_a"] + links["lat_b"]) / 2).to_numpy()
    mid_lon = ((links["lon_a"] + links["lon_b"]) / 2).to_numpy()
    dist = common.haversine_km(
        mid_lat[:, None],
        mid_lon[:, None],
        stations["lat"].to_numpy()[None, :],
        stations["lon"].to_numpy()[None, :],
    )
    near = dist <= GAUGE_RADIUS_KM
    stn_ids = stations["stn"].to_numpy()
    links["ref_ids"] = [";".join(str(s) for s in stn_ids[m]) for m in near]
    links["ref_dist_km"] = [
        float(d[m].mean()) if m.any() else np.nan
        for d, m in zip(dist, near, strict=True)
    ]
    cand = links[
        links["freq_ghz"].between(*FREQ_GHZ)
        & links["length_km"].between(*LENGTH_KM)
        & (links["ref_ids"] != "")
    ].copy()
    cand_ids = sorted(cand["cml_id"].unique())

    parts: dict[str, list[pd.Series]] = {k: [] for k in cand["link_id"]}
    levels: dict[str, list[np.ndarray]] = {k: [] for k in cand["link_id"]}
    flags: dict[str, int] = {}
    n_samples = 0
    for f in files:
        with xr.open_dataset(f) as ds:
            n_samples += ds.sizes["time"]
            pos = {str(c): i for i, c in enumerate(ds["cml_id"].values)}
            present = [c for c in cand_ids if c in pos]
            idx = np.array([pos[c] for c in present], dtype=int)
            # Stamps mark the interval end; label by interval start.
            start = pd.DatetimeIndex(ds["time"].values) - pd.Timedelta(
                minutes=SERIES_MINUTES
            )
            # Whole arrays are read once (the files are small and compressed);
            # indexing the netCDF by a scattered label list is far slower.
            order = ("cml_id", "sublink_id", "time")
            lo = ds["rsl_min"].transpose(*order).values[idx].astype(float)
            hi = ds["rsl_max"].transpose(*order).values[idx].astype(float)
            freq = ds["frequency"].transpose("cml_id", "sublink_id").values[idx]
            sub = [str(s) for s in ds["sublink_id"].values]
        a_lo, c_lo = clean_levels(None, lo, RSL_FLOOR_DBM, RSL_CEILING_DBM)
        a_hi, c_hi = clean_levels(None, hi, RSL_FLOOR_DBM, RSL_CEILING_DBM)
        for cnt in (c_lo, c_hi):
            for k, v in cnt.items():
                flags[k] = flags.get(k, 0) + v
        # Pmin above Pmax cannot happen in a valid interval.
        with np.errstate(invalid="ignore"):
            swapped = lo > hi
        flags["min_gt_max"] = flags.get("min_gt_max", 0) + int(swapped.sum())
        attn = (a_lo + a_hi) / 2
        attn[swapped] = np.nan
        for i, c in enumerate(present):
            for j, s in enumerate(sub):
                lid = f"{c}_{s}"
                if lid not in parts:
                    continue
                f0 = cand.loc[cand["link_id"] == lid, "freq_ghz"].iloc[0]
                if not abs(freq[i, j] - f0) < 0.01:
                    continue  # a different radio under the same path label
                parts[lid].append(pd.Series(attn[i, j], index=start))
                levels[lid].append(np.r_[lo[i, j], hi[i, j]])

    avail = {
        k: sum(int(s.notna().sum()) for s in v) / n_samples for k, v in parts.items()
    }
    cand["availability"] = cand["link_id"].map(avail)
    cand = cand[cand["availability"] >= MIN_AVAILABILITY]
    # One sublink per link, the better available one; nearest gauges first.
    cand = cand.sort_values(
        ["availability", "link_id"], ascending=[False, True]
    ).drop_duplicates("cml_id")
    chosen = cand.sort_values(["ref_dist_km", "link_id"]).head(MAX_SUBLINKS)
    chosen = chosen.sort_values("link_id").reset_index(drop=True)

    series = pd.concat(
        [
            pd.DataFrame(
                {"link_id": lid, "time": s.index, "attn_db": s.to_numpy()}
            ).dropna()
            for lid in chosen["link_id"]
            for s in parts[lid]
        ],
        ignore_index=True,
    )

    ref_parts = []
    for row in chosen.itertuples():
        ids = [int(s) for s in str(row.ref_ids).split(";")]
        r = rain[rain["stn"].isin(ids)].groupby("time")["rain_mmh"].mean()
        ref_parts.append(
            pd.DataFrame(
                {"link_id": row.link_id, "time": r.index, "rain_mmh": r.to_numpy()}
            )
        )
    reference = pd.concat(ref_parts, ignore_index=True)
    lo_t, hi_t = series["time"].min(), series["time"].max()
    reference = reference[reference["time"].between(lo_t, hi_t)]
    reference = reference.sort_values(["link_id", "time"]).reset_index(drop=True)

    chosen["pol"] = "V"
    chosen["ref_kind"] = "gauge"
    chosen["attn_kind"] = "-rsl"
    chosen["quant_db"] = [
        quant_step(np.concatenate(levels[k])) for k in chosen["link_id"]
    ]

    meta = {
        "dataset": NAME,
        "ref_minutes": REF_MINUTES,
        "series_minutes": SERIES_MINUTES,
        "start": str(series["time"].min()),
        "end": str(series["time"].max()),
        "n_links": len(chosen),
        "selection": (
            f"All four months ({MONTHS[0]} to {MONTHS[-1]}). Sublinks with "
            f"{FREQ_GHZ[0]:g}-{FREQ_GHZ[1]:g} GHz, path length "
            f"{LENGTH_KM[0]:g}-{LENGTH_KM[1]:g} km, a KNMI hourly station with "
            f"data within {GAUGE_RADIUS_KM:g} km of the link midpoint, and at "
            f"least {MIN_AVAILABILITY:.0%} valid 15-min samples over the four "
            "months. One sublink per link, the one with more valid samples; "
            f"the {MAX_SUBLINKS} links nearest to their gauge."
        ),
        "sources": [str(f.relative_to(root)) for f in files]
        + [str(g.relative_to(root)) for g in gauge_files],
        "flagged_samples": flags,
        "vendors": chosen["vendor"].value_counts().to_dict(),
        "notes": (
            "series.parquet is at the native 15-min sampling (series_minutes), "
            "not 1 min, labelled by interval start (archive stamp - 15 min, "
            "UTC). attn_db = -(Pmin + Pmax) / 2 in dB, the negative mid-range "
            "received level; there is no transmitted level, so its absolute "
            "value is arbitrary. Pmin and Pmax at or below "
            f"{RSL_FLOOR_DBM:g} dBm (receiver floor and the NEC sentinel "
            f"-99.9) or at or above {RSL_CEILING_DBM:g} dBm (sentinels 0 and "
            "-1), Pmin > Pmax, and single-sample jumps of more than 40 dB that "
            "revert are NaN and dropped. quant_db is per link: 1 dB for Nokia, "
            "0.1 dB for NEC. pol is V for all links because the archive does "
            "not give it (mostly vertical). rain_mmh is the KNMI hourly sum "
            "(0.1 mm units, -1 read as 0) in mm/h, the mean over the stations "
            "within the radius, labelled by hour start (HH - 1, UT). No gauge "
            "file exists for May 2012, so May has CML data and no reference."
        ),
    }
    common.write(out, chosen, series, reference, meta)
    return out


if __name__ == "__main__":
    print(build())
