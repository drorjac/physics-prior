"""Built CML datasets: schema and a positive attenuation-rain correlation.

The data never enter the repository, so each test is skipped unless that
dataset has been built with its `build()` under the CML data root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physprior.cml.sources import common
from physprior.cml.sources.openrainer import drop_spikes, quant_step

DATASETS = ("openrainer", "netherlands")


def _built(name: str):
    return pytest.mark.skipif(
        not (common.out_dir(name) / "meta.json").exists(),
        reason=f"{name} not built",
    )


def _bin_corr(d: dict, link_id: str) -> float:
    """Correlation of bin-mean (attn - 24 h rolling median) with reference rain."""
    meta = d["meta"]
    step = int(meta.get("series_minutes", 1))
    rm = int(meta["ref_minutes"])
    s = d["series"]
    s = s[s["link_id"] == link_id].set_index("time")["attn_db"].sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq=f"{step}min")
    full = s.reindex(grid).astype(float)
    win = 24 * 60 // step
    x = (
        (full - full.rolling(win, min_periods=win // 4, center=True).median())
        .resample(f"{rm}min")
        .mean()
    )
    r = d["reference"]
    r = r[r["link_id"] == link_id].set_index("time")["rain_mmh"]
    j = pd.concat([x, r], axis=1, sort=True).dropna()
    return float(j.corr().iloc[0, 1])


@pytest.mark.parametrize("name", [pytest.param(n, marks=_built(n)) for n in DATASETS])
def test_schema(name):
    d = common.load(name)
    links, series, ref, meta = d["links"], d["series"], d["reference"], d["meta"]
    common.validate(links, series, ref, meta)
    assert meta["dataset"] == name
    assert 1 <= len(links) <= 40
    assert links["freq_ghz"].between(15, 40).all()
    assert links["length_km"].between(0.3, 20).all()
    gauge = links["ref_kind"] == "gauge"
    assert (links.loc[gauge, "ref_dist_km"] <= 3.0).all()
    assert series["time"].dt.tz is None and ref["time"].dt.tz is None
    step = pd.Timedelta(minutes=int(meta.get("series_minutes", 1)))
    assert (series["time"].dt.floor(step) == series["time"]).all()
    rm = pd.Timedelta(minutes=int(meta["ref_minutes"]))
    assert (ref["time"].dt.floor(rm) == ref["time"]).all()
    assert np.isfinite(series["attn_db"]).all()
    assert not series.duplicated(["link_id", "time"]).any()
    assert not ref.duplicated(["link_id", "time"]).any()
    wet = float((ref["rain_mmh"] > 0.1).mean())
    assert 0.005 < wet < 0.5


@pytest.mark.parametrize("name", [pytest.param(n, marks=_built(n)) for n in DATASETS])
def test_attenuation_follows_rain(name):
    d = common.load(name)
    tot = d["reference"].groupby("link_id")["rain_mmh"].sum().nlargest(3)
    corr = [_bin_corr(d, k) for k in tot.index]
    assert min(corr) > 0.3, corr


def test_drop_spikes_flags_only_reverting_jumps():
    a = np.array([10.0, 10.0, 60.0, 10.0, 10.0, 55.0, 56.0, 57.0])
    m = drop_spikes(a)
    assert m.tolist() == [False, False, True, False, False, False, False, False]


def test_quant_step():
    rng = np.random.default_rng(0)
    assert quant_step(rng.integers(-60, -40, 1000).astype(float)) == 1.0
    assert quant_step(np.round(rng.normal(-50, 2, 5000), 1)) == 0.1
