"""OpenMesh (New York City): received levels of a community mesh network.

Archive: `openmesh/cml/ssid_to_links_os.nc` (rsl only, 1 dB steps, sampled
every 8 s until mid 2024 and every 6 min or coarser after), its link table
`openmesh/cml/metadata.csv` (MHz and metres), the quality-controlled Weather
Underground stations `openmesh/weather/pws_wu_merged_..._qc.nc` and the
ASOS 1-min record `openmesh/weather/asos_....nc`. All times are UTC.

The archive is sparse: most (cml, sublink) slots are empty and each slot is
stored as one chunk over the full time axis. Candidate slots are therefore
chosen from the link table first and read one at a time.

Period. Only the months whose time axis is sampled at least once a minute on
most days are eligible (October 2023 to June 2024). Within them the six
consecutive calendar months with the most rain, as the mean of the PWS
network, are kept.

Reference. In the QC'd PWS file `rainfall_amount` is already the amount per
report: the QC differentiated the cumulative daily totals (which reset at
local midnight) and dropped stations it could not repair. A report at time t
is the amount over (t - dt, t], with dt the spacing to the previous report;
reports after a gap longer than 16 min are dropped because their amount
cannot be placed in time. ASOS rows are 1-min amounts ending at their time
stamp. Each instrument becomes a 1-min rate in mm/h, the instruments within
3 km of a link midpoint are averaged, and the result is averaged over 15-min
bins that are at least two-thirds covered. Bins where the mean air
temperature at those instruments is below 2 C are set to NaN, since tipping
buckets misreport snow and the attenuation of snow differs from that of rain.
Before any of this, a gauge is dropped when its hourly rain over the period
correlates with the network median at r < 0.7 or its total differs from the
median's by more than a factor of two (12 of 77 gauges in December 2023 to
May 2024, among them two whose daily totals are shifted by a day).

Selection. Sublinks of 0.05-20 km with a known frequency, at least one
instrument within 3 km and data in at least 30 % of the network-wide wet
15-min bins. Slots whose series duplicate one already chosen (the same radio
listed under two cml ids) are skipped. All 24 GHz and 60 GHz sublinks that
qualify are kept. The 5 GHz sublinks are left out (`MAX_LOW_BAND = 0`): at
5 GHz the attenuation of most rain is below the 1 dB quantisation step, so
the link carries almost no rain signal.

Attenuation is -rsl averaged from 8 s to 1 min, so its level is arbitrary.
Levels of 0 dBm or above -20 dBm are logging sentinels and levels below
-100 dBm are impossible for these radios; both are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_root, haversine_km, out_dir, write

DATASET = "openmesh"
MAX_LINKS = 40
MAX_LOW_BAND = 0  # 5 GHz links are left out: rain is mostly below their 1 dB step
MAX_DIST_KM = 3.0
MIN_LEN_KM, MAX_LEN_KM = 0.05, 20.0
REF_MINUTES = 15
MIN_WET_AVAIL = 0.3
MONTHS = 6
MAX_GAP_MIN = 16
MIN_TEMP_C = 2.0
MIN_GAUGE_CORR = 0.7
RSL_MIN, RSL_MAX = -100.0, -20.0
WET_MMH = 0.1


def _paths() -> dict[str, Path]:
    d = data_root() / DATASET
    w = d / "weather"
    return {
        "cml": d / "cml" / "ssid_to_links_os.nc",
        "meta": d / "cml" / "metadata.csv",
        "pws": w / "pws_wu_merged_2023-06-07_2026-04-24_qc.nc",
        "asos": w / "asos_2023-10-01_2026-04-23.nc",
    }


def _cml_time(path: Path) -> pd.DatetimeIndex:
    import netCDF4

    with netCDF4.Dataset(path) as d:
        v = d["time"]
        require_ns = "nanoseconds" in v.units
        t = np.asarray(v[:], dtype="int64")
    return pd.DatetimeIndex(
        t.astype("datetime64[ns]" if require_ns else "datetime64[s]")
    )


def _read_instruments(
    p: dict[str, Path], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1-min rain rate (mm/h), temperature (C) and positions of every gauge."""
    import netCDF4

    grid = pd.date_range(start, end, freq="1min", inclusive="left")
    rate: dict[str, np.ndarray] = {}
    temp: dict[str, np.ndarray] = {}
    pos = []

    def put(
        times: pd.DatetimeIndex, values: np.ndarray, width: np.ndarray
    ) -> np.ndarray:
        # value covers the `width` minutes ending at its time stamp
        out = np.full(len(grid), np.nan)
        reps = width.astype(int)
        ends = np.repeat(times.values, reps)
        offs = np.concatenate([np.arange(k, 0, -1) for k in reps] + [np.zeros(0, int)])
        mins = (ends - offs.astype("timedelta64[m]")).astype("datetime64[m]")
        idx = (mins - grid.values[0].astype("datetime64[m]")).astype(int)
        ok = (idx >= 0) & (idx < len(grid))
        out[idx[ok]] = np.repeat(values, reps)[ok]
        return out

    with netCDF4.Dataset(p["pws"]) as d:
        for name, g in d.groups.items():
            if not name.startswith("KNYNEWYO") or "rainfall_amount" not in g.variables:
                continue
            if getattr(g, "qc_status", "") == "dropped":
                continue
            t = pd.to_datetime(np.asarray(g["time"][:]), unit="s")
            keep = (t > start - pd.Timedelta(minutes=MAX_GAP_MIN)) & (t <= end)
            if keep.sum() < 1000:
                continue
            t = t[keep]
            a = np.ma.filled(g["rainfall_amount"][0, :], np.nan)[keep].astype(float)
            gap = np.diff(t.values).astype("timedelta64[s]").astype(float) / 60.0
            gap = np.concatenate([[np.nan], gap])
            ok = np.isfinite(a) & (gap >= 1) & (gap <= MAX_GAP_MIN) & (a >= 0)
            width = np.rint(gap[ok])
            r = a[ok] / (gap[ok] / 60.0)
            r[r > 300] = np.nan  # spikes beyond any recorded NYC rain rate
            rate[name] = put(t[ok], r, width)
            if "temperature" in g.variables:
                tc = np.ma.filled(g["temperature"][0, :], np.nan)[keep].astype(float)
                temp[name] = put(t[ok], tc[ok], width)
            pos.append((name, "pws", float(g["lat"][0]), float(g["lon"][0])))

    with netCDF4.Dataset(p["asos"]) as d:
        for name, g in d.groups.items():
            t = pd.to_datetime(np.asarray(g["time"][:]), unit="s")
            keep = (t > start) & (t <= end)
            t = t[keep]
            a = np.ma.filled(g["rainfall_amount"][0, :], np.nan)[keep].astype(float)
            a[(a < 0) | (a > 5)] = np.nan  # fill values and > 300 mm/h minutes
            tc = np.ma.filled(g["temperature"][0, :], np.nan)[keep].astype(float)
            tc[(tc < -50) | (tc > 50)] = np.nan
            one = np.ones(len(t))
            rate["ASOS_" + name] = put(t, a * 60.0, one)
            temp["ASOS_" + name] = put(t, tc, one)
            pos.append(("ASOS_" + name, "asos", float(g["lat"][0]), float(g["lon"][0])))

    return (
        pd.DataFrame(rate, index=grid),
        pd.DataFrame(temp, index=grid),
        pd.DataFrame(pos, columns=["id", "kind", "lat", "lon"]).set_index("id"),
    )


def _consistent(rate: pd.DataFrame) -> list[str]:
    """Gauges whose hourly rain agrees with the network median.

    A gauge is kept when its hourly series correlates with the median of all
    gauges at r >= MIN_GAUGE_CORR and its total over common hours is within a
    factor of two of the median's. This removes stations with shifted clocks,
    blocked funnels or unrepaired counters that the archive QC let through.
    """
    h = rate.resample("h").mean()
    med = h.median(axis=1)
    keep = []
    for c in h.columns:
        j = pd.concat([h[c], med], axis=1).dropna()
        if len(j) < 24 * 14 or j.iloc[:, 1].sum() <= 0:
            continue
        r = float(j.corr().iloc[0, 1])
        ratio = float(j.iloc[:, 0].sum() / j.iloc[:, 1].sum())
        if r >= MIN_GAUGE_CORR and 0.5 <= ratio <= 2.0:
            keep.append(str(c))
    return keep


def _choose_period(time: pd.DatetimeIndex, p: dict[str, Path]) -> pd.Timestamp:
    """First month of the wettest run of MONTHS months with 1-min sampling."""
    import netCDF4

    per_day = pd.Series(time.floor("min").unique()).dt.floor("D").value_counts()
    dense_day = per_day >= 1000
    month_ok = dense_day.groupby(dense_day.index.to_period("M")).mean() >= 0.9
    totals = []
    with netCDF4.Dataset(p["pws"]) as d:
        for name, g in d.groups.items():
            if not name.startswith("KNYNEWYO") or "rainfall_amount" not in g.variables:
                continue
            if getattr(g, "qc_status", "") == "dropped":
                continue
            t = pd.to_datetime(np.asarray(g["time"][:]), unit="s")
            a = pd.Series(np.ma.filled(g["rainfall_amount"][0, :], np.nan), index=t)
            totals.append(a.resample("MS").sum(min_count=1000))
    rain = pd.concat(totals, axis=1).mean(axis=1)
    rain.index = rain.index.to_period("M")
    best, best_sum = None, -1.0
    months = sorted(month_ok.index)
    for i in range(len(months) - MONTHS + 1):
        run = months[i : i + MONTHS]
        if run[-1] != run[0] + MONTHS - 1 or not all(month_ok[m] for m in run):
            continue
        s = float(rain.reindex(run).sum())
        if s > best_sum:
            best, best_sum = run[0], s
    if best is None:
        raise RuntimeError("no run of months with 1-min CML sampling")
    return best.to_timestamp()


def _candidates(meta: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["key"] = m["cml_id"].astype(str) + "|" + m["sublink_id"]
    m["freq_ghz"] = m["frequency"] / 1000.0
    m["length_km"] = m["length"] / 1000.0
    m = m[
        (m["length_km"] >= MIN_LEN_KM)
        & (m["length_km"] <= MAX_LEN_KM)
        & (m["freq_ghz"] >= 1)
        & (m["freq_ghz"] <= 100)
    ].copy()
    mid_lat = ((m["site_0_lat"] + m["site_1_lat"]) / 2).to_numpy()[:, None]
    mid_lon = ((m["site_0_lon"] + m["site_1_lon"]) / 2).to_numpy()[:, None]
    d = haversine_km(
        mid_lat, mid_lon, inst["lat"].to_numpy()[None], inst["lon"].to_numpy()[None]
    )
    names = inst.index.to_numpy()
    near = d <= MAX_DIST_KM
    m["ref_ids"] = [";".join(names[row]) for row in near]
    m["ref_dist_km"] = [
        float(dr[row].mean()) if row.any() else np.nan
        for dr, row in zip(d, near, strict=True)
    ]
    m["ref_kind"] = [
        "+".join(sorted(set(inst.loc[list(names[row]), "kind"]))) if row.any() else ""
        for row in near
    ]
    return m[m["ref_ids"] != ""].set_index("key")


def _reference(rate: pd.DataFrame, temp: pd.DataFrame, ids: list[str]) -> pd.Series:
    r = rate[ids]
    bins = r.resample(f"{REF_MINUTES}min", label="left", closed="left")
    cover = bins.count() / REF_MINUTES
    per = bins.mean().where(cover >= 2 / 3)
    ref = per.mean(axis=1, skipna=True)
    tc = temp.reindex(columns=[i for i in ids if i in temp.columns])
    tbin = (
        tc.resample(f"{REF_MINUTES}min", label="left", closed="left")
        .mean()
        .mean(axis=1)
    )
    return ref.where(~(tbin < MIN_TEMP_C))


def build(out: Path | None = None) -> Path:
    import netCDF4

    p = _paths()
    time = _cml_time(p["cml"])
    start = _choose_period(time, p)
    end = start + pd.DateOffset(months=MONTHS)
    rate, temp, inst = _read_instruments(p, start, end)
    good_ids = _consistent(rate)
    rate, inst = rate[good_ids], inst.loc[good_ids]
    meta = pd.read_csv(p["meta"])
    cand = _candidates(meta, inst)

    net = rate[[c for c in rate.columns if not c.startswith("ASOS")]].mean(axis=1)
    net15 = net.resample(f"{REF_MINUTES}min", label="left", closed="left").mean()
    wet_bins = net15.index[net15 > WET_MMH]

    in_period = np.flatnonzero((time >= start) & (time < end))
    i0, i1 = int(in_period[0]), int(in_period[-1]) + 1
    tt = time[i0:i1]
    series_by_key: dict[str, pd.Series] = {}
    avail: dict[str, float] = {}
    seen: set[bytes] = set()
    with netCDF4.Dataset(p["cml"]) as d:
        cml_ids = [str(c) for c in d["cml_id"][:]]
        sub_ids = [str(s) for s in d["sublink_id"][:]]
        for key, row in cand.iterrows():
            ci, si = cml_ids.index(str(row["cml_id"])), sub_ids.index(row["sublink_id"])
            x = np.ma.filled(d["rsl"][ci, si, i0:i1].astype("float64"), np.nan)
            x[(x < RSL_MIN) | (x > RSL_MAX)] = np.nan
            ok = np.isfinite(x)
            if ok.sum() < 1000:
                continue
            digest = x[ok][:20000].tobytes() + np.flatnonzero(ok)[:20000].tobytes()
            if digest in seen:
                continue
            seen.add(digest)
            s = pd.Series(-x[ok], index=tt[ok])
            a1 = s.groupby(s.index.floor("min")).mean()
            has = (
                a1.resample(f"{REF_MINUTES}min").count().reindex(wet_bins).fillna(0) > 0
            )
            avail[str(key)] = float(has.mean())
            if avail[str(key)] >= MIN_WET_AVAIL:
                series_by_key[str(key)] = a1

    good = cand.loc[list(series_by_key)].copy()
    good["avail"] = pd.Series(avail)
    good = good.sort_values("avail", ascending=False)
    high = good[good["freq_ghz"] >= 10]
    low = good[good["freq_ghz"] < 10].head(min(MAX_LOW_BAND, MAX_LINKS - len(high)))
    sel = pd.concat([high, low]).head(MAX_LINKS)

    series_parts, ref_parts = [], []
    for key, row in sel.iterrows():
        lid = str(key).replace("|", "_")
        a1 = series_by_key[str(key)]
        series_parts.append(
            pd.DataFrame(
                {"link_id": lid, "time": a1.index.values, "attn_db": a1.values}
            )
        )
        ref = _reference(rate, temp, row["ref_ids"].split(";")).dropna()
        ref_parts.append(
            pd.DataFrame(
                {"link_id": lid, "time": ref.index.values, "rain_mmh": ref.values}
            )
        )
    series = pd.concat(series_parts, ignore_index=True)
    reference = pd.concat(ref_parts, ignore_index=True)
    series["time"] = series["time"].astype("datetime64[ns]")
    reference["time"] = reference["time"].astype("datetime64[ns]")

    links = pd.DataFrame(
        {
            "link_id": [str(k).replace("|", "_") for k in sel.index],
            "freq_ghz": sel["freq_ghz"].to_numpy(float),
            "pol": "V",
            "length_km": sel["length_km"].to_numpy(float),
            "lat_a": sel["site_0_lat"].to_numpy(float),
            "lon_a": sel["site_0_lon"].to_numpy(float),
            "lat_b": sel["site_1_lat"].to_numpy(float),
            "lon_b": sel["site_1_lon"].to_numpy(float),
            "ref_kind": sel["ref_kind"].to_numpy(),
            "ref_ids": sel["ref_ids"].to_numpy(),
            "ref_dist_km": sel["ref_dist_km"].to_numpy(float),
            "attn_kind": "-rsl",
            "quant_db": 1.0,
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
            "ref_minutes": REF_MINUTES,
            "start": str(series["time"].min().date()),
            "end": str(series["time"].max().date()),
            "n_links": len(links),
            "selection": (
                f"the {MONTHS} wettest consecutive months with 1-min CML sampling; "
                f"sublinks of {MIN_LEN_KM:g}-{MAX_LEN_KM:g} km with a gauge within "
                f"{MAX_DIST_KM:g} km of the midpoint and data in at least "
                f"{MIN_WET_AVAIL:.0%} of the wet {REF_MINUTES}-min bins, duplicates "
                f"removed; every 24 and 60 GHz sublink, then at most {MAX_LOW_BAND} "
                "5 GHz sublinks by wet-bin availability"
            ),
            "sources": [str(v) for v in p.values()],
            "notes": (
                "attn = -RSL, 8 s averaged to 1 min, level arbitrary, 1 dB steps; "
                "RSL of 0 or above -20 dBm (sentinels) and below -100 dBm dropped. "
                "reference = mean of the 1-min rain rates of the QC'd PWS (per-report "
                "amounts over the interval since the previous report, gaps above "
                f"{MAX_GAP_MIN} min dropped) and ASOS 1-min gauges within "
                f"{MAX_DIST_KM:g} km, averaged over {REF_MINUTES}-min bins at least "
                f"two-thirds covered; bins with mean air temperature below "
                f"{MIN_TEMP_C:g} C set to NaN (snow). Link ids are cml_sublink."
            ),
        },
    )
    return out
