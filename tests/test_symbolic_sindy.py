"""SINDy, and the derivative studies it ships with (CONTRIBUTING rule 5)."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.symbolic.sindy import (
    damped_oscillator,
    derivative_convergence,
    fit_sindy,
    library,
    noise_amplification,
    pendulum,
    score_model,
    simulate,
    stlsq,
)


@pytest.mark.parametrize("make", [damped_oscillator, pendulum])
def test_sindy_recovers_the_equations_from_clean_trajectories(make):
    sys = make()
    t, X = simulate(sys)
    m = fit_sindy(
        t, X, sys.state, threshold=0.05, method="savgol", window=11, trig=sys.trig
    )
    s = score_model(m, sys)
    assert s["support_ok"] and s["coef_rel_err"] < 1e-4


def test_stlsq_zeroes_small_terms():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    Theta, labels = library(X, ("a", "b"), degree=2)
    dX = (2.0 * X[:, 0] - 0.5 * X[:, 0] * X[:, 1])[:, None]
    Xi = stlsq(Theta, dX, threshold=0.1)
    nz = {labels[i] for i in np.flatnonzero(Xi[:, 0])}
    assert nz == {"a", "a*b"}


def test_central_differences_converge_at_second_order():
    rows = derivative_convergence()
    orders = [r["fd_observed_order"] for r in rows if "fd_observed_order" in r]
    assert all(abs(o - 2.0) < 0.05 for o in orders)


def test_finite_difference_noise_follows_sigma_over_dt():
    rows = [r for r in noise_amplification(sigmas=(1e-2,)) if r["method"] == "fd"]
    ratio = np.mean([r["rms_err"] / r["fd_theory"] for r in rows])
    assert ratio == pytest.approx(1.0, rel=0.1)
    sg = [r for r in noise_amplification(sigmas=(1e-2,)) if r["method"] == "savgol"]
    assert np.mean([r["rms_err"] for r in sg]) < 0.3 * np.mean(
        [r["rms_err"] for r in rows]
    )
