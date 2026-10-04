"""The one format every CML dataset is reduced to before modelling.

Each source module (`openmrg`, `openrainer`, `netherlands`, `openmesh`) has a
`build(out_dir)` that reads the archive under the CML data root and writes
four files to `out_dir` (by default `<CML_DATA_ROOT>/<dataset>/derived/hybrid`):

    links.csv          one row per selected sublink (columns: LINK_COLUMNS)
    series.parquet     link_id, time, attn_db         (1-min, UTC, naive)
    reference.parquet  link_id, time, rain_mmh        (bin start, UTC, naive)
    meta.json          META_KEYS

`attn_db` is the total attenuation in dB: TSL - RSL where the archive has a
transmitted level, -RSL where it has only the received level (then its level
is arbitrary, and only changes in it mean anything). Missing minutes are NaN
or absent. Nothing is baseline-corrected here; that is the model's job.

`rain_mmh` is the reference path-averaged rain rate in mm/h over the bin
[time, time + ref_minutes). It is the mean of the reference instruments the
source chose for that link (gauges within some distance, or radar pixels
along the path); `links.csv` says which. A bin with no valid reference is
absent or NaN.

The data never enter the repository; only the code that builds them does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from physprior.units import require

LINK_COLUMNS = (
    "link_id",  # str, unique within the dataset
    "freq_ghz",  # carrier frequency, GHz
    "pol",  # "H" or "V"
    "length_km",  # path length, km
    "lat_a",
    "lon_a",
    "lat_b",
    "lon_b",
    "ref_kind",  # e.g. "gauge", "radar", "pws"
    "ref_ids",  # ";"-joined ids of the reference instruments used
    "ref_dist_km",  # distance from link midpoint to the reference (mean), km
    "attn_kind",  # "tsl-rsl" or "-rsl"
    "quant_db",  # quantisation step of the signal levels, dB (0 if none)
)

META_KEYS = (
    "dataset",
    "ref_minutes",  # reference bin length, minutes (>= 1)
    "start",  # ISO date of the first sample
    "end",  # ISO date of the last sample
    "n_links",
    "selection",  # how links were chosen, in words
    "sources",  # list of archive files read
    "notes",  # anything a reader must know (units, quirks)
)


def data_root() -> Path:
    return Path(os.environ.get("CML_DATA_ROOT", Path.home() / "data/cml"))


def out_dir(dataset: str) -> Path:
    return data_root() / dataset / "derived" / "hybrid"


def write(
    out: Path,
    links: pd.DataFrame,
    series: pd.DataFrame,
    reference: pd.DataFrame,
    meta: dict,
) -> None:
    """Validate and write the four files."""
    validate(links, series, reference, meta)
    out.mkdir(parents=True, exist_ok=True)
    links.loc[:, list(LINK_COLUMNS)].to_csv(out / "links.csv", index=False)
    s = series.loc[:, ["link_id", "time", "attn_db"]].copy()
    s["attn_db"] = s["attn_db"].astype("float32")
    s.to_parquet(out / "series.parquet", index=False)
    r = reference.loc[:, ["link_id", "time", "rain_mmh"]].copy()
    r["rain_mmh"] = r["rain_mmh"].astype("float32")
    r.to_parquet(out / "reference.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def validate(
    links: pd.DataFrame, series: pd.DataFrame, reference: pd.DataFrame, meta: dict
) -> None:
    missing = [c for c in LINK_COLUMNS if c not in links.columns]
    require(not missing, f"links.csv lacks {missing}")
    missing = [k for k in META_KEYS if k not in meta]
    require(not missing, f"meta.json lacks {missing}")
    require(links["link_id"].is_unique, "link_id must be unique")
    require(set(links["pol"]) <= {"H", "V"}, "pol must be H or V")
    f = links["freq_ghz"].to_numpy(float)
    require(bool(np.all((f >= 1) & (f <= 100))), "freq_ghz outside 1-100 GHz")
    L = links["length_km"].to_numpy(float)
    require(bool(np.all((L > 0.01) & (L < 60))), "length_km outside (0.01, 60)")
    ids = set(links["link_id"])
    require(set(series["link_id"].unique()) <= ids, "series has unknown links")
    require(set(reference["link_id"].unique()) <= ids, "reference has unknown links")
    require(
        np.issubdtype(series["time"].dtype, np.datetime64), "series.time not datetime"
    )
    require(
        np.issubdtype(reference["time"].dtype, np.datetime64),
        "reference.time not datetime",
    )
    rr = reference["rain_mmh"].to_numpy(float)
    rr = rr[np.isfinite(rr)]
    require(bool(np.all(rr >= 0)), "negative rain rate in reference")
    require(bool(np.all(rr < 500)), "rain rate above 500 mm/h in reference")
    require(int(meta["ref_minutes"]) >= 1, "ref_minutes must be >= 1")


def load(dataset: str, root: Path | None = None) -> dict:
    """The four files of one built dataset."""
    d = root or out_dir(dataset)
    require((d / "meta.json").exists(), f"{dataset} not built: run its build()")
    return {
        "links": pd.read_csv(d / "links.csv", dtype={"link_id": str}),
        "series": pd.read_parquet(d / "series.parquet"),
        "reference": pd.read_parquet(d / "reference.parquet"),
        "meta": json.loads((d / "meta.json").read_text()),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
