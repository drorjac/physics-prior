"""The optimization study's tasks, models, curvature tools and runner."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.optim import optimizers_study as O  # noqa: E402
from physprior.optim import problems as Pb  # noqa: E402


def test_oscillator_solution_satisfies_its_ode():
    """Central-difference residual of the closed form falls as h^2."""
    g, w = Pb.OSC_TRUTH["gamma"], Pb.OSC_TRUTH["omega0"]
    t = np.linspace(0.5, 9.5, 50)
    errs = []
    for h in (1e-2, 5e-3):
        x = [Pb.oscillator_solution(t + k * h, g, w) for k in (-1, 0, 1)]
        xt = (x[2] - x[0]) / (2 * h)
        xtt = (x[2] - 2 * x[1] + x[0]) / h**2
        errs.append(np.max(np.abs(xtt + 2 * g * xt + w**2 * x[1])))
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)
    assert Pb.oscillator_solution(np.array([0.0]), g, w)[0] == pytest.approx(1.0)


def test_overdamped_branch_is_the_cosh_solution():
    g, w, t = 3.0, 2.0, np.linspace(0, 3, 7)
    k = np.sqrt(g**2 - w**2)
    ref = np.exp(-g * t) * (np.cosh(k * t) + g / k * np.sinh(k * t))
    assert np.allclose(Pb.oscillator_solution(t, g, w), ref)
    tt = torch.tensor(t).reshape(-1, 1)
    got = Pb._osc_law_t(tt, torch.tensor(g), torch.tensor(w)).numpy()
    assert np.allclose(got, ref)


def test_heat_solution_satisfies_its_pde():
    h = Pb.HEAT_TRUTH
    x, t = np.linspace(0.1, 0.9, 9), 0.2
    errs = []
    for d in (1e-3, 5e-4):
        u = lambda xx, tt: Pb.heat_solution(xx, tt, **h)  # noqa: E731
        ut = (u(x, t + d) - u(x, t - d)) / (2 * d)
        uxx = (u(x + d, t) - 2 * u(x, t) + u(x - d, t)) / d**2
        errs.append(np.max(np.abs(ut - h["D"] * uxx)))
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)


def test_physics_model_at_truth_sits_at_the_noise_floor():
    task = Pb.oscillator()
    m = Pb.Model(task, "physics", seed=0)
    with torch.no_grad():
        for spec, r in zip(m.ps.specs, m.ps.raw, strict=True):
            r.fill_(float(np.log(task.truth[spec.name] / spec.init)))
        d = float(m.data_loss())
    assert d == pytest.approx((Pb.OSC_NOISE / task.sd_y) ** 2, rel=0.5)
    assert d < task.tol


def test_full_hessian_of_a_quadratic():
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    p = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    H = Pb.full_hessian(lambda: 0.5 * p @ A @ p, [p])
    assert np.allclose(H, A.numpy())


def test_lanczos_matches_the_full_spectrum():
    task = Pb.oscillator(n_train=12, n_colloc=10)
    m = Pb.Model(task, "nn", seed=0)
    eig = np.linalg.eigvalsh(Pb.full_hessian(m.loss, m.parameters()))[::-1]
    top = Pb.top_eigenvalues(m.loss, m.parameters(), k=3)
    assert np.allclose(top, eig[:3], rtol=1e-6)
    s = Pb.spectrum_summary(eig)
    assert s["n_eig"] == m.n_params()
    assert 0 < s["frac_constrained"] <= 1


@pytest.mark.parametrize("model", Pb.MODELS)
@pytest.mark.parametrize("opt", ["adam", "lbfgs"])
def test_run_one_returns_a_scored_row(model, opt):
    spec = O.RunSpec(
        "oscillator", model, opt, 1e-2 if opt == "adam" else 1.0, 3, 40, record=True
    )
    row, curve, _ = O.run_one(spec)
    assert row["evals_used"] >= 1
    assert {"nrmse_in", "nrmse_out", "final_data_loss", "reached_tol"} <= set(row)
    assert curve and curve[0]["evals"] == 0
    if model == "pinn":
        assert np.isfinite(curve[0]["grad_phys"])


def test_levenberg_marquardt_reaches_tolerance_on_the_oscillator():
    row = O.run_lm("oscillator", 11)
    assert row["reached_tol"]
    assert row["param_err_pct"] < 10.0


def test_lr_selection_ignores_diverged_runs():
    import pandas as pd

    g = pd.DataFrame(
        {
            "task": ["a"] * 4,
            "model": ["nn"] * 4,
            "optimizer": ["sgd"] * 4,
            "lr": [0.1, 0.1, 1.0, 1.0],
            "final_data_loss": [1e-2, 2e-2, 1e-5, 1e-5],
            "diverged": [False, False, True, True],
        }
    )
    assert float(O.select_lr(g).lr.iloc[0]) == 0.1
