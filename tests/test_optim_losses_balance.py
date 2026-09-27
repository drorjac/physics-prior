"""Loss functions, the w_phys dial, and the `balance` diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.methods.pinn import FROZEN_PINN, PhysParam, fit_pinn  # noqa: E402
from physprior.optim import losses_study as L  # noqa: E402
from physprior.optim import pinn_balance as B  # noqa: E402


def test_robust_losses_are_quadratic_near_zero():
    r = torch.tensor([1e-3, -2e-3], dtype=torch.float64)
    q = float(torch.mean(0.5 * r**2))
    assert float(L.robust_loss("huber", r, 1.0)) == pytest.approx(q)
    assert float(L.robust_loss("cauchy", r, 1.0)) == pytest.approx(q, rel=1e-5)
    assert float(L.robust_loss("l1", r, 1.0)) == pytest.approx(1.5e-3)


def test_huber_is_linear_in_the_tails():
    s = 0.1
    d = L.HUBER_K * s
    a = float(L.robust_loss("huber", torch.tensor([10.0], dtype=torch.float64), s))
    b = float(L.robust_loss("huber", torch.tensor([20.0], dtype=torch.float64), s))
    assert b - a == pytest.approx(10.0 * d)


def test_outlier_noise_displaces_the_stated_fraction():
    t, y = L.make_data("outliers", np.random.default_rng(0))
    from physprior.optim.problems import OSC_TRUTH, oscillator_solution

    big = np.abs(y - oscillator_solution(t, **OSC_TRUTH)) > 0.35
    assert big.sum() == round(L.OUTLIER_FRAC * L.N_TRAIN)


def test_loss_cell_runs():
    rows = L._cell(("physics", "gaussian", 11, 0, 60))
    assert [r["loss"] for r in rows] == list(L.LOSSES)
    assert all(np.isfinite(r["err_gamma_pct"]) for r in rows)


def test_w_dial_cell_runs():
    row = L._w_cell(("gaussian", 1.0, 11, 20, 1e-2))
    assert np.isfinite(row["nrmse_in"]) and row["w_phys"] == 1.0


def _toy():
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 4.0, 30).reshape(-1, 1)
    return x, 3.0 * x.ravel() ** 1.5 + rng.normal(0.0, 0.05, 30)


def test_history_recovers_the_annealed_weight():
    """w = (total - data) / phys from the stored history reproduces the
    weight the fit reports, and under `balance` it grows from its start."""
    x, y = _toy()
    f = fit_pinn(
        x,
        y,
        lambda xt, GM: GM * xt.squeeze(-1) ** 1.5,
        [PhysParam("GM", 2.0)],
        epochs=801,
        seed=3,
        options=FROZEN_PINN,
        record_every=100,
    )
    h = B.history_frame(f)
    assert h.w_phys.iloc[-1] == pytest.approx(f.extra["w_phys_final"], rel=1e-6)
    assert h.w_phys.iloc[-1] > 1.0
    assert h.correction_rms_frac.iloc[-1] < h.correction_rms_frac.iloc[0]


def test_feedback_slope_of_a_pure_runaway_is_minus_one():
    import pandas as pd

    ep = np.arange(0, 1000, 100.0)
    corr = 1e-2 / (1 + ep)
    hist = pd.DataFrame(
        {
            "track": "t",
            "split": "s",
            "seed": 3,
            "variant": "balanced",
            "epoch": ep,
            "w_phys": 1.0 / corr,
            "correction_rms_frac": corr,
            "data_loss": 0.0,
        }
    )
    fb = B.feedback_slopes(hist)
    assert fb.slope_logw_vs_logcorr.iloc[0] == pytest.approx(-1.0)


@pytest.mark.network
def test_every_balance_track_loads_and_splits():
    for tr in B.TRACKS:
        prob = B.load_problem(tr)
        for itr, ite in B.splits(tr, prob, 3).values():
            assert len(np.intersect1d(itr, ite)) == 0
            assert len(itr) >= 3 and len(ite) >= 1
