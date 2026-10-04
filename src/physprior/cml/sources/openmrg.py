"""OpenMRG (Gothenburg, June-August 2015): 10-s TSL and RSL, city rain gauges.

Archive: `openmrg/cml/openmrg_cml_full.nc` (SMHI, CC BY-SA 4.0,
doi:10.5281/zenodo.6673750), its metadata CSV (already in GHz and km), and
the ten city gauges `openmrg/weather/gauges/city/CityGauges-2015JJA.csv`
(1-min accumulations in mm, 0.1 or 0.2 mm resolution). All times are UTC.

Selection, as in Jacoby et al. (ICASSP 2026): vertically polarised sublinks
at 28-30 and 38-40 GHz, here those whose midpoint lies within 3 km of at
least one city gauge, the closest 40. Attenuation is TSL - RSL averaged from
10 s to 1 min. The reference is the mean 1-min rain rate of every gauge
within 3 km of the midpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_root, haversine_km, out_dir, write

DATASET = "openmrg"
MAX_LINKS = 40
MAX_DIST_KM = 3.0
BANDS = ((28.0, 30.0), (38.0, 40.0))


def _paths() -> dict[str, Path]:
    d = data_root() / DATASET
    return {
        "cml": d / "cml" / "openmrg_cml_full.nc",
        "meta": d / "cml" / "openmrg_cml_metadata.csv",
        "gauges": d / "weather" / "gauges" / "city" / "CityGauges-2015JJA.csv",
        "gauge_meta": d / "weather" / "gauges" / "city" / "CityGauges-metadata.csv",
    }


def select_links(meta: pd.DataFrame, gmeta: pd.DataFrame) -> pd.DataFrame:
    """Sublinks in the paper's bands with a gauge within 3 km of the midpoint."""
    m = meta.copy()
    in_band = np.zeros(len(m), bool)
    for lo, hi in BANDS:
        in_band |= (m["Frequency_GHz"] >= lo) & (m["Frequency_GHz"] <= hi)
    m = m[in_band & (m["Polarization"].str.upper().str[0] == "V")].copy()
    m["mid_lat"] = (m["NearLatitude_DecDeg"] + m["FarLatitude_DecDeg"]) / 2
    m["mid_lon"] = (m["NearLongitude_DecDeg"] + m["FarLongitude_DecDeg"]) / 2
    d = haversine_km(
        m["mid_lat"].to_numpy()[:, None],
        m["mid_lon"].to_numpy()[:, None],
        gmeta["Latitude_DecDeg"].to_numpy()[None, :],
        gmeta["Longitude_DecDeg"].to_numpy()[None, :],
    )
    names = gmeta["Name"].to_numpy()
    m["ref_ids"] = [";".join(names[row <= MAX_DIST_KM]) for row in d]
    m["ref_dist_km"] = [
        float(row[row <= MAX_DIST_KM].mean()) if np.any(row <= MAX_DIST_KM) else np.nan
        for row in d
    ]
    m = m[m["ref_ids"] != ""]
    return m.sort_values("ref_dist_km").head(MAX_LINKS)


def build(out: Path | None = None) -> Path:
    import xarray as xr

    p = _paths()
    meta = pd.read_csv(p["meta"])
    gmeta = pd.read_csv(p["gauge_meta"], encoding="utf-8-sig")
    sel = select_links(meta, gmeta)

    ds = xr.open_dataset(p["cml"])
    idx = [int(np.flatnonzero(ds["sublink"].values == s)[0]) for s in sel["Sublink"]]
    tsl = ds["tsl"].isel(sublink=idx).values.astype("float64")
    rsl = ds["rsl"].isel(sublink=idx).values.astype("float64")
    time = pd.DatetimeIndex(ds["time"].values)
    ds.close()

    # sentinel and impossible values: received above transmitted, or levels
    # outside any physical range for these radios
    bad = (rsl > tsl) | (rsl < -100) | (rsl > 0) | (tsl < -30) | (tsl > 40)
    attn = np.where(bad, np.nan, tsl - rsl)
    a10 = pd.DataFrame(attn, index=time, columns=sel["Sublink"].astype(str))
    a1 = a10.resample("1min", label="left", closed="left").mean()
    series = (
        a1.rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="link_id", value_name="attn_db")
    )

    g = pd.read_csv(p["gauges"])
    g["time"] = pd.to_datetime(g["Time_UTC"]).dt.tz_localize(None)
    # each row is the accumulation over the minute ending at its time stamp,
    # so the bin starts one minute earlier
    g["time"] = g["time"] - pd.Timedelta(minutes=1)
    g = g.set_index("time").drop(columns="Time_UTC") * 60.0  # mm/min -> mm/h
    refs = []
    for _, row in sel.iterrows():
        cols = row["ref_ids"].split(";")
        r = g[cols].mean(axis=1, skipna=True)
        refs.append(
            pd.DataFrame(
                {"link_id": str(row["Sublink"]), "time": r.index, "rain_mmh": r.values}
            )
        )
    reference = pd.concat(refs, ignore_index=True)

    links = pd.DataFrame(
        {
            "link_id": sel["Sublink"].astype(str).to_numpy(),
            "freq_ghz": sel["Frequency_GHz"].to_numpy(float),
            "pol": "V",
            "length_km": sel["Length_km"].to_numpy(float),
            "lat_a": sel["NearLatitude_DecDeg"].to_numpy(float),
            "lon_a": sel["NearLongitude_DecDeg"].to_numpy(float),
            "lat_b": sel["FarLatitude_DecDeg"].to_numpy(float),
            "lon_b": sel["FarLongitude_DecDeg"].to_numpy(float),
            "ref_kind": "gauge",
            "ref_ids": sel["ref_ids"].to_numpy(),
            "ref_dist_km": sel["ref_dist_km"].to_numpy(float),
            "attn_kind": "tsl-rsl",
            "quant_db": 0.3,
        }
    )
    out = out or out_dir(DATASET)
    write(
        out,
        links,
        series,
        reference,
        {
            "dataset": DATASET,
            "ref_minutes": 1,
            "start": str(series["time"].min().date()),
            "end": str(series["time"].max().date()),
            "n_links": len(links),
            "selection": (
                "V-pol sublinks at 28-30 and 38-40 GHz with a city gauge within "
                f"{MAX_DIST_KM:g} km of the midpoint, the closest {MAX_LINKS}"
            ),
            "sources": [str(v) for v in p.values()],
            "notes": (
                "attn = TSL - RSL, 10 s averaged to 1 min; reference = mean of the "
                "1-min gauge accumulations within 3 km, times 60, bin start = gauge "
                "time stamp minus 1 min"
            ),
        },
    )
    return out
