"""NOAA Integrated Surface Database (ISD), ISD-Lite hourly files.

Two products from NCEI:

  isd-history.csv    the station list: USAF and WBAN identifiers, name,
                     country, latitude and longitude (decimal degrees),
                     elevation (m above sea level), period of record.
  isd-lite/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz
                     one year of hourly observations per station, reduced
                     from the full ISD record to eight variables.

ISD-Lite is fixed width, one observation per line (Technical Document,
NCEI, "ISD-Lite format", 2006). Columns, 1-based:

     1-4   year                 14-19  air temperature, tenths of degC
     6-7   month                20-25  dew point, tenths of degC
     9-10  day                  26-31  sea-level pressure, tenths of hPa
    12-13  hour (UTC)           32-61  wind, sky cover, precipitation

Missing values are -9999. The loader checks the fixed-width columns on a
sample of lines and then splits on whitespace, so a change of format raises
rather than shifting a column.

The snapshot this module builds is a MONTHLY MEAN of the observations at one
UTC hour, per station. The rule is set by `SnapshotRule` and fixed before any
fit is run; see `problems/fields/weather.py`.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.exceptions import DownloadError
from physprior.units import require

from ..cache import cached_get, provenance

HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
HISTORY_FILE = "isd_history.csv"
LITE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{sid}-{year}.gz"
MISSING = -9999

# Near-surface air temperature anywhere on Earth lies within these bounds
# (record low -89.2 degC, Vostok 1983; record high 56.7 degC, Death Valley
# 1913, WMO archive). A value outside them is a decoding error.
T_MIN_C, T_MAX_C = -90.0, 60.0


# ---------------------------------------------------------------------------
# station list
# ---------------------------------------------------------------------------


def load_station_list() -> tuple[pd.DataFrame, dict]:
    """The ISD station history, with numeric coordinates. Returns (table, prov).

    Columns: sid ("USAF-WBAN"), name, ctry, lat, lon, elev_m, begin, end
    (begin/end as YYYYMMDD integers). Rows with no position are dropped.
    """
    path = cached_get(HISTORY_URL, HISTORY_FILE)
    raw = pd.read_csv(path, dtype=str)
    need = {"USAF", "WBAN", "STATION NAME", "CTRY", "LAT", "LON", "ELEV(M)"}
    require(need <= set(raw.columns), f"ISD history: missing columns {need}")
    df = pd.DataFrame(
        {
            "sid": raw["USAF"].str.strip() + "-" + raw["WBAN"].str.strip(),
            "name": raw["STATION NAME"].fillna("").str.strip(),
            "ctry": raw["CTRY"].fillna("").str.strip(),
            "lat": pd.to_numeric(raw["LAT"], errors="coerce"),
            "lon": pd.to_numeric(raw["LON"], errors="coerce"),
            "elev_m": pd.to_numeric(raw["ELEV(M)"], errors="coerce"),
            "begin": pd.to_numeric(raw["BEGIN"], errors="coerce"),
            "end": pd.to_numeric(raw["END"], errors="coerce"),
        }
    ).dropna(subset=["lat", "lon", "begin", "end"])
    require(len(df) > 10000, f"ISD history: only {len(df)} stations parsed")
    require(
        bool(df.lat.between(-90, 90).all() and df.lon.between(-180, 180).all()),
        "ISD history: coordinates are not decimal degrees",
    )
    return df.reset_index(drop=True), provenance(path, HISTORY_URL)


def stations_in_box(
    stations: pd.DataFrame, box: tuple[float, float, float, float], year: int
) -> pd.DataFrame:
    """Stations inside (lon_min, lon_max, lat_min, lat_max) that report for
    the whole of `year` and have a physical elevation.

    -999.9 is the list's missing-elevation code; a land station in a
    mountain box is above the Dead Sea shore (-430 m) and below 5000 m.
    """
    lon0, lon1, lat0, lat1 = box
    sel = (
        stations.lon.between(lon0, lon1)
        & stations.lat.between(lat0, lat1)
        & (stations.begin <= year * 10000 + 101)
        & (stations.end >= year * 10000 + 1231)
        & stations.elev_m.between(-430.0, 5000.0)
    )
    return stations[sel].reset_index(drop=True)


# ---------------------------------------------------------------------------
# hourly files
# ---------------------------------------------------------------------------


# (start, stop) of year, month, day, hour and air temperature, 0-based
_COLS = ((0, 4), (5, 7), (8, 10), (11, 13), (13, 19))


def parse_lite(text: str, year: int) -> pd.DataFrame:
    """ISD-Lite text -> month, day, hour, temp_c (NaN when missing)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    require(len(lines) > 0, "ISD-Lite: empty file")
    # The fixed-width layout is checked on a sample of lines; the bulk parse
    # then splits on whitespace, which is equivalent when the check holds.
    for ln in lines[:24] + lines[-24:]:
        require(len(ln) >= 61, f"ISD-Lite: short line {ln!r}")
        f = ln.split()
        require(len(f) == 12, f"ISD-Lite: expected 12 fields in {ln!r}")
        for k, (c0, c1) in enumerate(_COLS):
            require(
                int(ln[c0:c1]) == int(f[k]),
                f"ISD-Lite: field {k} not in columns {c0 + 1}-{c1}: {ln!r}",
            )
    a = np.array([ln.split()[:5] for ln in lines], dtype=np.int64)
    require(bool(np.all(a[:, 0] == year)), f"ISD-Lite: rows not all from {year}")
    require(bool(np.all((a[:, 1] >= 1) & (a[:, 1] <= 12))), "ISD-Lite: bad month")
    require(bool(np.all((a[:, 2] >= 1) & (a[:, 2] <= 31))), "ISD-Lite: bad day")
    require(bool(np.all((a[:, 3] >= 0) & (a[:, 3] <= 23))), "ISD-Lite: bad hour")
    t = a[:, 4].astype(float)
    t[a[:, 4] == MISSING] = np.nan
    t /= 10.0  # tenths of degC -> degC
    ok = np.isfinite(t)
    require(
        bool(np.all((t[ok] >= T_MIN_C) & (t[ok] <= T_MAX_C))),
        "ISD-Lite: air temperature outside [-90, 60] degC; not tenths of degC?",
    )
    return pd.DataFrame(
        {"month": a[:, 1], "day": a[:, 2], "hour": a[:, 3], "temp_c": t}
    )


def _missing_manifest(year: int):
    return get_settings().raw_dir / f"isd_lite_{year}_missing.json"


def load_station_year(sid: str, year: int) -> tuple[pd.DataFrame, dict] | None:
    """One station-year of hourly temperatures, or None if NCEI has no file.

    Stations listed as active can still have no ISD-Lite file for a year.
    Those are remembered in a small manifest in the raw directory so a rerun
    does not ask the server again for a file it has already said is absent.
    """
    manifest = _missing_manifest(year)
    missing = set(json.loads(manifest.read_text())) if manifest.exists() else set()
    if sid in missing:
        return None
    url = LITE_URL.format(year=year, sid=sid)
    try:
        path = cached_get(url, f"isd_lite_{sid}-{year}.gz")
    except DownloadError as exc:
        if "404" in str(exc):
            missing.add(sid)
            manifest.write_text(json.dumps(sorted(missing)))
            return None
        raise
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    return parse_lite(text, year), provenance(path, url)


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotRule:
    """How one station value is formed. Fixed before any fit is run.

    The value is the mean over `month` of `year` of the observations at
    `hour` UTC. A station enters if it has such an observation on at least
    `min_coverage` of the days of the month. Two records at the same site
    (within `dedup_deg` in latitude and longitude and `dedup_m` in
    elevation) are one station; the one with more observations is kept.
    """

    year: int
    month: int
    hour: int = 12
    min_coverage: float = 0.8
    box: tuple[float, float, float, float] = (5.8, 13.5, 45.5, 48.0)
    dedup_deg: float = 0.02
    dedup_m: float = 50.0

    def describe(self) -> str:
        lon0, lon1, lat0, lat1 = self.box
        return (
            f"mean of {self.hour:02d} UTC air temperature over "
            f"{self.year}-{self.month:02d}, stations with a reading on at "
            f">= {self.min_coverage:.0%} of days, box lon {lon0}..{lon1}, "
            f"lat {lat0}..{lat1}"
        )


@dataclass
class StationField:
    """One scalar field sampled at stations."""

    sid: np.ndarray
    name: np.ndarray
    lon: np.ndarray  # degrees east
    lat: np.ndarray  # degrees north
    elev_m: np.ndarray  # m above sea level
    temp_c: np.ndarray  # degC
    n_days: np.ndarray  # days contributing to the mean
    rule: SnapshotRule
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.temp_c)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sid": self.sid,
                "name": self.name,
                "lon": self.lon,
                "lat": self.lat,
                "elev_m": self.elev_m,
                "temp_c": self.temp_c,
                "n_days": self.n_days,
            }
        )


def _days_in_month(year: int, month: int) -> int:
    return int(pd.Period(f"{year}-{month:02d}").days_in_month)


def _dedupe(df: pd.DataFrame, rule: SnapshotRule) -> pd.DataFrame:
    df = df.sort_values(["n_days", "sid"], ascending=[False, True])
    keep: list[int] = []
    for i, r in df.iterrows():
        dup = any(
            abs(r.lat - df.at[j, "lat"]) < rule.dedup_deg
            and abs(r.lon - df.at[j, "lon"]) < rule.dedup_deg
            and abs(r.elev_m - df.at[j, "elev_m"]) < rule.dedup_m
            for j in keep
        )
        if not dup:
            keep.append(i)
    return df.loc[keep].sort_values("sid").reset_index(drop=True)


def monthly_snapshot(rule: SnapshotRule) -> StationField:
    """Build the station field for `rule`. Downloads on a cold cache."""
    stations, hist_prov = load_station_list()
    cand = stations_in_box(stations, rule.box, rule.year)
    require(len(cand) >= 30, f"ISD: only {len(cand)} candidate stations in box")
    n_month = _days_in_month(rule.year, rule.month)
    rows, files = [], []
    for st in cand.itertuples():
        got = load_station_year(st.sid, rule.year)
        if got is None:
            continue
        obs, prov = got
        files.append(prov)
        sel = obs[
            (obs.month == rule.month) & (obs.hour == rule.hour) & obs.temp_c.notna()
        ]
        n_days = int(sel.day.nunique())
        if n_days < rule.min_coverage * n_month:
            continue
        rows.append(
            {
                "sid": st.sid,
                "name": st.name,
                "lon": float(st.lon),
                "lat": float(st.lat),
                "elev_m": float(st.elev_m),
                # one reading per day at a fixed hour, so this is a daily mean
                "temp_c": float(sel.groupby("day").temp_c.mean().mean()),
                "n_days": n_days,
            }
        )
    require(len(rows) >= 30, f"ISD: only {len(rows)} stations pass the rule")
    df = _dedupe(pd.DataFrame(rows), rule)
    require(
        bool(df.temp_c.between(-40.0, 45.0).all()),
        "ISD: a monthly mean outside [-40, 45] degC",
    )
    require(float(df.elev_m.max()) > 1500.0, "ISD: no mountain stations in the box")
    prov = {
        "station_list": hist_prov,
        "hourly_files": len(files),
        "hourly_bytes": int(sum(p["bytes"] for p in files)),
        "hourly_url_pattern": LITE_URL,
        "hourly": files,
        "rule": asdict(rule) | {"text": rule.describe()},
        "candidates": len(cand),
        "stations": len(df),
    }
    return StationField(
        sid=df.sid.to_numpy(),
        name=df.name.to_numpy(),
        lon=df.lon.to_numpy(float),
        lat=df.lat.to_numpy(float),
        elev_m=df.elev_m.to_numpy(float),
        temp_c=df.temp_c.to_numpy(float),
        n_days=df.n_days.to_numpy(int),
        rule=rule,
        provenance=prov,
    )
