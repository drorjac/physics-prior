"""The built OpenMesh dataset: schema, time base and rain signal.

Skipped when `physprior.cml.sources.openmesh.build()` has not been run on this
machine, since the archive is not part of the repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physprior.cml.sources import common

DATASET = "openmesh"

pytestmark = pytest.mark.skipif(
    not (common.out_dir(DATASET) / "meta.json").exists(),
    reason="OpenMesh hybrid dataset not built",
)


@pytest.fixture(scope="module")
def data() -> dict:
    return common.load(DATASET)


def test_schema(data):
    links, series, ref, meta = (
        data["links"],
        data["series"],
        data["reference"],
        data["meta"],
    )
    common.validate(links, series, ref, meta)
    assert list(links.columns) == list(common.LINK_COLUMNS)
    assert 1 <= len(links) <= 40
    assert meta["n_links"] == len(links)
    assert set(links["attn_kind"]) == {"-rsl"}
    assert np.allclose(links["quant_db"], 1.0)
    assert links["length_km"].between(0.05, 20).all()
    assert (links["ref_dist_km"] <= 3).all()
    assert (links["freq_ghz"] > 20).any(), "no 24 or 60 GHz link selected"


def test_times_are_utc_naive_minutes(data):
    for frame in (data["series"], data["reference"]):
        t = frame["time"]
        assert t.dtype.kind == "M"
        assert t.dt.tz is None
    s = data["series"]["time"]
    assert (s.dt.second == 0).all()
    r = data["reference"]["time"]
    assert (r.dt.minute % data["meta"]["ref_minutes"] == 0).all()
    span = s.max() - s.min()
    assert span <= pd.Timedelta(days=190)


def test_reference_is_rain(data):
    r = data["reference"]["rain_mmh"].to_numpy(float)
    wet = float(np.mean(r > 0.1))
    assert 0.01 < wet < 0.3


def test_attenuation_tracks_rain(data):
    """Excess attenuation correlates positively with the reference on the
    highest-frequency links, where rain attenuation exceeds the 1 dB step."""
    links, series, ref = data["links"], data["series"], data["reference"]
    rm = int(data["meta"]["ref_minutes"])
    high = links.loc[links["freq_ghz"] > 50, "link_id"]
    corrs = []
    for lid in high:
        a = series.loc[series["link_id"] == lid].set_index("time")["attn_db"]
        a = a.reindex(pd.date_range(a.index.min(), a.index.max(), freq="min"))
        excess = a - a.rolling(1440, min_periods=60).median()
        eb = excess.resample(f"{rm}min").mean()
        r = ref.loc[ref["link_id"] == lid].set_index("time")["rain_mmh"]
        j = pd.concat([eb, r], axis=1).dropna()
        corrs.append(float(j.corr().iloc[0, 1]))
    assert corrs, "no link above 50 GHz"
    assert np.median(corrs) > 0.3
    assert min(corrs) > 0
