"""reconstruction.fields: the series, the FD solver, the free-space potentials."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physprior.exceptions import UnitError
from physprior.reconstruction import convergence
from physprior.reconstruction import fields as F


@pytest.mark.parametrize("d,n", [(1, 128), (2, 32)])
def test_fd_solver_agrees_with_series(d, n):
    f = F.draw_field("box", d, 3)
    g1, u = F.solve_fd(d, n, f.source, lambda z: f.geom.lift(z) @ f.lift_coef)
    X = np.stack(np.meshgrid(*([g1] * d), indexing="ij"), -1).reshape(-1, d)
    us = f.value(X)
    assert np.max(np.abs(u.ravel() - us)) / np.std(us) < 0.05


def test_fd_solver_is_second_order_on_manufactured_solutions():
    df = convergence.fd_manufactured(quick=True)
    orders = df.dropna(subset=["observed_order"])
    last = orders.groupby(["d", "boundary"]).observed_order.last()
    assert np.all((last > 1.8) & (last < 2.3))


def test_series_fit_truncation_is_small():
    df = convergence.series_truncation()
    at_fit = df[df.is_fit_kmax]
    assert (at_fit.rel_max_err < 1e-4).all()


@pytest.mark.parametrize("d", [1, 2, 3])
def test_free_potential_solves_poisson(d):
    df = convergence.free_residual()
    g = df[df.d == d].sort_values("h")
    assert g.rel_max_residual.iloc[0] < 3e-4
    assert 1.8 < g.observed_order.dropna().iloc[-1] < 2.2


@pytest.mark.parametrize("d", [1, 2, 3])
def test_free_potential_tends_to_point_source(d):
    r = np.array([0.8, 1.2])
    phi = F.free_potential(r, d, 0.05)
    point = {1: -0.5 * r, 2: -np.log(r) / (2 * np.pi), 3: 1 / (4 * np.pi * r)}[d]
    # the 1-D Gaussian potential differs from -|x|/2 by a vanishing tail only
    np.testing.assert_allclose(phi, point, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_box_green_gradient_matches_finite_differences(d):
    x = np.random.default_rng(0).uniform(0, 1, (20, d))
    c = np.full(d, 0.4)
    _, g = F.box_green(x, c, w=0.07, km=12, grad=True)
    h = 1e-6
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        fd = (
            F.box_green(x, c + e, w=0.07, km=12)[0]
            - F.box_green(x, c - e, w=0.07, km=12)[0]
        ) / (2 * h)
        np.testing.assert_allclose(g[:, j], fd, atol=1e-6)


@pytest.mark.parametrize("case", F.CASES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_candidate_basis_matches_direct_evaluation(case, d):
    geom = F.Geometry(case, d, km=10)
    x = np.random.default_rng(1).uniform(0, 1, (30, d))
    B, pos = geom.candidates(x, 3)
    direct = np.column_stack([geom.green(x, p)[0] for p in pos])
    np.testing.assert_allclose(B, direct, atol=1e-12)


def test_fields_are_reproducible_and_seeded():
    a, b = F.draw_field("box", 2, 11), F.draw_field("box", 2, 11)
    np.testing.assert_array_equal(a.centers, b.centers)
    assert not np.allclose(a.centers, F.draw_field("box", 2, 23).centers)
    lo, hi = F.SOURCE_REGION
    assert np.all((a.centers >= lo) & (a.centers <= hi))


def test_sensor_designs_are_nested():
    f = F.draw_field("free", 2, 11)
    x1, y1, _ = F.sensors(f, 16, 11, 0.01, 1.0)
    x2, y2, _ = F.sensors(f, 64, 11, 0.01, 1.0)
    np.testing.assert_array_equal(x1, x2[:16])
    np.testing.assert_array_equal(y1, y2[:16])


def test_mismatch_fields_differ_from_the_model():
    base = F.draw_field("box", 2, 11)
    X = F.eval_grid(base.geom, 21)
    extra = F.mismatch_field("extra_source", 11)
    assert len(extra.extra_q) == 1
    assert np.std(extra.value(X) - base.value(X)) > 1e-3
    neu = F.mismatch_field("neumann_wall", 11)
    assert np.std(neu.value(X) - base.value(X)) > 1e-3


def test_guards():
    with pytest.raises(UnitError):
        F.Geometry("box", 4)
    with pytest.raises(UnitError):
        F.Geometry("torus", 2)
    f = F.draw_field("box", 2, 3)
    with pytest.raises(UnitError):
        f.value(np.zeros((3, 3)))
    with pytest.raises(UnitError):
        F.sensors(f, 4, 3, 1.5, 1.0)


def test_truth_truncation_is_converged():
    f = F.draw_field("box", 3, 3)
    x = np.random.default_rng(2).uniform(0, 1, (50, 3))
    g = replace(f.geom, km=F.KMAX_TRUTH[3] - 6)
    v = g.lift(x) @ f.lift_coef + sum(
        q * g.green(x, c)[0] for c, q in zip(f.centers, f.q, strict=True)
    )
    assert np.max(np.abs(v - f.value(x))) < 1e-8
