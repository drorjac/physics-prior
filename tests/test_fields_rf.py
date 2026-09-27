"""fields/rf: the Helmholtz solver, the law, the inverse fit, the track's plumbing.

The solver tests use a small scene so they run in seconds; the full run is
marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from physprior.benchmark.protocol import Problem
from physprior.exceptions import UnitError
from physprior.problems.fields import rf
from physprior.problems.fields import rf_sim as S

SMALL = S.Scene("small", width=6.0, height=5.0, tx=(2.5, 2.5))


def _hankel_db_error(ppw: int) -> float:
    fld = S.solve(SMALL, ppw)
    X, Y = np.meshgrid(fld.x, fld.y)
    r = np.hypot(X - SMALL.tx[0], Y - SMALL.tx[1])
    m = fld.interior_mask(margin=0.0) & (r >= 1.5)
    exact = S.analytic_free_space(r[m])
    err = 10 * np.log10(np.abs(fld.u[m]) ** 2 / np.abs(exact) ** 2)
    return float(np.sqrt(np.mean(err**2)))


def test_free_space_matches_the_hankel_function():
    # PML and discretisation together: a reflecting edge would leave a
    # standing wave far above this
    assert _hankel_db_error(12) < 0.3


def test_solver_error_falls_as_h_squared():
    e8, e16 = _hankel_db_error(8), _hankel_db_error(16)
    assert 3.0 < e8 / e16 < 5.5


def test_solver_rejects_an_absurd_resolution():
    with pytest.raises(UnitError):
        S.solve(SMALL, 2)


def test_concrete_is_lossy_with_the_right_sign():
    # exp(-i w t): loss is a positive imaginary permittivity
    assert S.EPS_CONCRETE.real == pytest.approx(5.24)
    assert 0.5 < S.EPS_CONCRETE.imag < 0.9


def test_a_wall_attenuates():
    size = {"width": 6.0, "height": 5.0, "tx": (1.5, 2.5)}
    wall = S.Scene("wall", blocks=(S.Box(3.5, 4.5, 0.0, 5.0),), **size)
    free = S.Scene("free", **size)
    a, b = S.solve(wall, 10), S.solve(free, 10)
    la, lb = a.local_mean_db(), b.local_mean_db()
    behind = a.at(la, np.array([5.0]), np.array([2.5])) - b.at(
        lb, np.array([5.0]), np.array([2.5])
    )
    assert behind[0] < -5.0


def test_local_mean_of_a_uniform_field_is_that_field():
    fld = S.solve(SMALL, 8)
    fld.u = np.full_like(fld.u, 0.1)
    lm = fld.local_mean_db()
    inner = fld.interior_mask()
    assert np.allclose(lm[inner], -20.0, atol=1e-9)


def test_free_space_law_is_the_large_r_limit():
    r = np.array([20.0, 50.0])
    exact = 10 * np.log10(np.abs(S.analytic_free_space(r)) ** 2)
    assert np.allclose(S.free_space_law_db(r), exact, atol=0.01)
    assert S.N_FREE_SPACE == 1.0


def test_choose_ppw_takes_the_coarsest_within_tolerance():
    rows = [
        {"ppw": 8, "local_mean_p99_db": 3.0},
        {"ppw": 12, "local_mean_p99_db": 0.4},
        {"ppw": 16, "local_mean_p99_db": 0.1},
        {"ppw": 24, "local_mean_p99_db": 0.0},
    ]
    assert S.choose_ppw(rows, 0.5) == 12
    assert S.choose_ppw(rows[:1] + rows[-1:], 0.5) == 24


# ---------------------------------------------------------------------------
# the law and the inverse problem
# ---------------------------------------------------------------------------


def test_law_numpy_and_torch_agree():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).uniform(0, 20, (50, 2))
    theta = {"P0": -28.0, "n": 1.3, "xt": 5.0, "yt": 7.0}
    a = rf.law_np(x, **theta)
    b = rf.law_t(torch.tensor(x), **{k: torch.tensor(v) for k, v in theta.items()})
    assert np.allclose(a, b.numpy())


def test_fit_law_finds_the_transmitter():
    scene = S.floor_plan()
    rng = np.random.default_rng(1)
    x = rng.uniform([1, 1], [23, 17], (120, 2))
    truth = {"P0": -29.0, "n": 1.0, "xt": 17.5, "yt": 4.2}
    y = rf.law_np(x, **truth) + rng.normal(0, 0.5, len(x))
    fit = rf.fit_law(scene, x, y)
    assert np.hypot(fit.params["xt"] - 17.5, fit.params["yt"] - 4.2) < 0.3
    assert fit.params["n"] == pytest.approx(1.0, abs=0.1)
    assert not any(fit.extra["at_bound"].values())


def test_receivers_stand_in_air_away_from_the_transmitter():
    scene = S.free_space()
    X, Y = np.meshgrid(np.linspace(0, 24, 97), np.linspace(0, 18, 73))
    ok = rf._valid(scene, X, Y)
    assert not np.any(ok & S.floor_plan().in_material(X, Y))
    r = np.hypot(X - scene.tx[0], Y - scene.tx[1])
    assert np.all(r[ok] >= rf.TX_EXCLUSION)


def test_extrapolation_split_is_the_transmitters_room():
    x = np.column_stack([np.linspace(1, 23, 40), np.full(40, 5.0)])
    prob = Problem(
        track="t",
        x=x,
        y=np.zeros(40),
        law_np=rf.law_np,
        law_t=None,
        params=[],
        theta_published={},
    )
    itr, ite = rf.extrapolation_split(prob)
    assert x[itr, 0].max() < rf.ROOM_EDGE < x[ite, 0].min()


def test_tutorial_cells_build():
    nbformat = pytest.importorskip("nbformat")
    from physprior.problems.fields.rf_tutorial import rf_cells

    cells = rf_cells()
    assert len(cells) >= 6
    assert all(isinstance(c, nbformat.NotebookNode) for c in cells)


@pytest.mark.slow
def test_quick_run_writes_results_and_a_doc(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from physprior.config import reset_settings

    monkeypatch.setenv("PHYSPRIOR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PHYSPRIOR_FIGURES_DIR", str(tmp_path / "figures"))
    reset_settings()
    try:
        meta = rf.run(quick=True)
        base = tmp_path / "results" / "fields" / "rf"
        for name in (
            "sweep_budget",
            "sweep_noise",
            "extrapolation",
            "grid_convergence",
        ):
            assert (base / f"{name}.csv").is_file()
        # the free-space law is complete: the oracle is within a dB of the truth
        assert meta["scenes"]["free"]["oracle_map_rmse_db"] < 1.0
        out = rf.render_doc(tmp_path / "rf.md")
        assert "Helmholtz" in (tmp_path / "rf.md").read_text()
        assert out.endswith("rf.md")
    finally:
        reset_settings()
