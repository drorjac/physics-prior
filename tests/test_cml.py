"""The CML study: ITU coefficients, the power law, samples, a tiny training."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from physprior.cml import sim  # noqa: E402
from physprior.cml.data import budget_mask, make_samples  # noqa: E402
from physprior.cml.itu import k_alpha  # noqa: E402
from physprior.cml.models import PowerLaw  # noqa: E402
from physprior.cml.train import TrainConfig, fit_arms, metrics  # noqa: E402

LINKS = pd.DataFrame(
    {
        "freq_ghz": [28.2, 38.5, 18.0],
        "pol": ["V", "V", "H"],
        "length_km": [1.0, 2.5, 4.0],
    }
)


def test_itu_table_values():
    # P.838-3 tabulated values at table points
    k, a = k_alpha(30.0, "V")
    assert k == pytest.approx(0.2291, rel=1e-3)
    assert a == pytest.approx(0.9129, rel=1e-3)
    k, a = k_alpha(20.0, "H")
    assert k == pytest.approx(0.09164, rel=1e-3)
    with pytest.raises(ValueError):
        k_alpha(150.0, "V")


def test_power_law_inverts_the_law():
    k, a = k_alpha(28.2, "V")
    L, r = 2.0, 12.0
    A = k * r**a * L
    pl = PowerLaw(np.array([k]), np.array([a]), np.array([L]), np.array([1e-6]), m=3)
    x = torch.full((1, 5), float(A))
    assert float(pl(x, torch.zeros(1, dtype=torch.long))) == pytest.approx(r, rel=1e-4)
    # below the dead zone nothing is rain
    pl0 = PowerLaw(np.array([k]), np.array([a]), np.array([L]), np.array([0.5]), m=3)
    assert float(pl0(torch.full((1, 5), 0.4), torch.zeros(1, dtype=torch.long))) == 0.0


@pytest.fixture(scope="module")
def tiny():
    cfg = sim.SimConfig("t", "ar1", True, n_links=3, days=12, seed=1)
    raw = sim.simulate(cfg, LINKS)
    return make_samples(**raw, history_min=30)


def test_samples_shape_and_split(tiny):
    s = tiny
    assert s.x.shape[1] == 30 and s.m == 10
    assert set(np.unique(s.split)) <= {0, 1, 2}
    # whole days go to one split
    for d in np.unique(s.day):
        assert len(np.unique(s.split[s.day == d])) == 1
    m = budget_mask(s, 0.3)
    assert m.sum() < (s.split == 0).sum()
    assert not np.any(m & (s.split != 0))


def test_metrics_definitions():
    r = np.array([0.0, 1.0, 3.0])
    out = metrics(r + 1.0, r)
    assert out["nbias"] == pytest.approx(1.0 / r.mean())
    assert out["nrmse"] == pytest.approx(1.0 / r.mean())


def test_all_arms_train(tiny):
    cfg = replace(TrainConfig().quick(), max_wet=500)
    out = fit_arms(tiny, 3, cfg=cfg)
    assert set(out["metrics"]) == {
        "pl_itu",
        "pl_cal",
        "gru",
        "hybrid_joint",
        "hybrid_gate",
        "hybrid_phased",
    }
    for v in out["metrics"].values():
        assert np.isfinite(v["nrmse"])
    g = out["gate"]["hybrid_phased"]
    assert np.all((g >= 0) & (g <= 1))
