"""reconstruction.methods: each method on a small field with known truth."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.reconstruction import fields as F
from physprior.reconstruction import methods as M


def _data(case, d, n, noise=0.0, seed=11):
    f = F.draw_field(case, d, seed)
    X = F.eval_grid(f.geom, {1: 101, 2: 25, 3: 9}[d])
    truth = f.value(X)
    x, y, _ = F.sensors(f, n, seed, noise, float(np.std(truth)))
    return f, X, truth, x, y


def _nrmse(fit, X, truth):
    return float(np.sqrt(np.mean((fit.predict(X) - truth) ** 2)) / np.std(truth))


@pytest.mark.parametrize("case,d,n", [("box", 1, 24), ("box", 2, 60), ("free", 2, 60)])
def test_physics_fit_recovers_sources_without_noise(case, d, n):
    f, X, truth, x, y = _data(case, d, n)
    fit = M.fit_physics(x, y, f.geom, seed=11)
    assert M.location_error(f.centers, fit.extra["centers"]) < 5e-3
    assert _nrmse(fit, X, truth) < 1e-2
    assert not fit.extra["at_bound"]


def test_gp_and_interp_learn_something():
    _, X, truth, x, y = _data("box", 2, 120, noise=0.01)
    for fit in (M.fit_gp(x, y), M.fit_interp(x, y, smoothing=1e-3)):
        assert _nrmse(fit, X, truth) < 0.3


def test_pigp_with_physics_mean_is_no_worse_than_physics():
    f, X, truth, x, y = _data("box", 2, 80, noise=0.01)
    ph = M.fit_physics(x, y, f.geom, seed=11)
    pg = M.fit_gp(x, y, mean=ph.predict, name="pigp")
    assert _nrmse(pg, X, truth) < 1.5 * _nrmse(ph, X, truth) + 1e-3


def test_location_error_matches_permutations():
    a = np.array([[0.1, 0.2], [0.7, 0.8]])
    assert M.location_error(a, a[::-1]) == pytest.approx(0.0)
    assert M.location_error(a, a + np.array([0.03, 0.04])) == pytest.approx(0.05)


def test_pinn_runs_and_returns_sources():
    pytest.importorskip("torch")
    f, X, _, x, y = _data("box", 1, 16, noise=0.01)
    fit = M.fit_pinn(x, y, f.geom, w_pde=1e-2, seed=3, epochs=30)
    assert fit.extra["centers"].shape == (3, 1)
    assert np.all(np.isfinite(fit.predict(X)))
