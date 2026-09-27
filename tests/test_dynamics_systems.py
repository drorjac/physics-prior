"""Reference solutions of the dynamics study: laws, invariants, convergence."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.dynamics import pdes as PD
from physprior.dynamics import systems as S


@pytest.mark.parametrize("name", list(S.SYSTEMS))
def test_reference_integrator_is_fourth_order_and_converged(name):
    c = S.convergence(S.get_system(name), n_ics=2)
    orders = [r["observed_order"] for r in c["rows"][1:]]
    assert np.nanmedian(orders) == pytest.approx(4.0, abs=0.6)
    assert c["used_max_err"] < 1e-6


def test_pendulum_and_kepler_conserve_their_invariants():
    for name, fns in (
        ("pendulum", [S.pendulum_H]),
        ("kepler", [S.kepler_E, S.kepler_L]),
    ):
        sys = S.get_system(name)
        u0 = sys.sample_ic(np.random.default_rng(0), 3, "test")
        tr = S.simulate(sys, u0, 100)
        for fn in fns:
            v = fn(tr)
            assert np.max(np.abs(v - v[:, :1])) < 1e-8


def test_pendulum_ood_energies_lie_above_the_training_range():
    sys = S.get_system("pendulum")
    rng = np.random.default_rng(1)
    h_tr = S.pendulum_H(sys.sample_ic(rng, 200, "train"))
    h_ood = S.pendulum_H(sys.sample_ic(rng, 200, "ood"))
    assert h_tr.max() <= 0.5 + 1e-9
    assert h_ood.min() >= 0.55 - 1e-9
    assert h_ood.max() < 1.0  # still below the separatrix


def test_kepler_initial_states_have_the_requested_orbit():
    sys = S.get_system("kepler")
    u = sys.sample_ic(np.random.default_rng(2), 50, "train")
    E = S.kepler_E(u)
    a = -1 / (2 * E)  # vis-viva, GM = 1
    assert np.all((a > 0.8 - 1e-9) & (a < 1.2 + 1e-9))


def test_known_parts_are_what_they_claim():
    u = np.array([[0.3, -0.2]])
    np.testing.assert_allclose(S.pendulum_known(u, S.NP), [[-0.2, -0.3]])
    v = np.array([[1.0, 2.0, 20.0]])
    diff = S.lorenz_f(v, S.NP) - S.lorenz_known(v, S.NP)
    np.testing.assert_allclose(diff, [[0.0, -1.0 * 20.0, 2.0]])


def test_heat_and_advection_are_exact():
    coef = PD.sample_ic(np.random.default_rng(0), 2, "test", "advection")
    period = round(2 * np.pi / (PD.C_ADV * 0.05))
    tr = PD.exact_linear("advection", coef, period, 2 * np.pi / period)
    np.testing.assert_allclose(tr[:, -1], tr[:, 0], atol=1e-10)
    h = PD.exact_linear("heat", coef, 10, 0.05)
    assert np.all(np.diff(h.var(axis=-1), axis=1) < 0)


def test_burgers_reference_matches_a_finer_solve():
    coef = PD.sample_ic(np.random.default_rng(3), 1, "test", "burgers")
    a = PD.burgers_reference(coef, 10, 0.05, n_fine=512, sub=25)
    b = PD.burgers_reference(coef, 10, 0.05, n_fine=1024, sub=50)
    assert np.max(np.abs(a - b)) < 1e-4
    # The solver conserves the mean exactly; the mean over the coarse sample
    # points differs only by the aliased content of steepening fronts.
    assert np.max(np.abs(a.mean(-1) - a[:, :1].mean(-1))) < 1e-6


def test_coarse_physics_step_is_consistent_with_the_reference():
    import torch

    coef = PD.sample_ic(np.random.default_rng(4), 2, "test", "heat")
    tr = PD.exact_linear("heat", coef, 5, PD.PDES["heat"].dt)
    step = PD.physics_step("heat")
    u = torch.as_tensor(tr[:, 0])
    for _ in range(5):
        u = step(u)
    assert np.max(np.abs(u.numpy() - tr[:, -1])) < 1e-2
