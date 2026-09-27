"""The learned steppers: shapes, structure, training and scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physprior.dynamics import metrics as M
from physprior.dynamics import pdes as PD
from physprior.dynamics import steppers as ST
from physprior.dynamics import systems as S
from physprior.dynamics import train as TR


@pytest.fixture(scope="module")
def pend():
    sys = S.get_system("pendulum")
    return sys, S.make_data(sys, 4, 3, n_test=2, n_val=1)


@pytest.mark.parametrize("arm", [*ST.ODE_ARMS, "oracle", "known_only"])
def test_every_ode_arm_steps_a_batch(pend, arm):
    sys, data = pend
    m = ST.build_ode(arm, sys, data)
    u = torch.as_tensor(data.test[:, 0])
    out = M.rollout(m, data.test[:, 0], 3, data.mean, data.std)
    assert out.shape == (2, 4, 2)
    assert np.isfinite(out).all()
    assert u.shape == (2, 2)


def test_oracle_stepper_reproduces_the_reference(pend):
    sys, data = pend
    m = ST.build_ode("oracle", sys, data)
    pred = M.rollout(m, data.test[:, 0], 20, data.mean, data.std)
    assert np.max(np.abs(pred - data.test[:, :21])) < 1e-4


def test_leapfrog_map_is_symplectic():
    """The Jacobian of the leapfrog update satisfies J^T Omega J = Omega for
    any network weights."""
    sys = S.get_system("pendulum")
    data = S.make_data(sys, 2, 7, n_test=1, n_val=1)
    m = ST.build_ode("hnn_leapfrog", sys, data)
    u = torch.tensor([0.4, -0.3], dtype=ST.DTYPE)
    J = torch.autograd.functional.jacobian(lambda x: m(x[None])[0], u)
    om = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=ST.DTYPE)
    np.testing.assert_allclose((J.T @ om @ J).detach().numpy(), om.numpy(), atol=1e-10)


def test_training_reduces_the_loss(pend):
    sys, data = pend
    m = ST.build_ode("residual", sys, data)
    r = TR.train(m, data.train, data.std, TR.TrainOptions(steps=60, batch=32), seed=3)
    assert r["loss"][-1] < r["loss"][0]
    r2 = TR.train(
        ST.build_ode("residual", sys, data),
        data.train,
        data.std,
        TR.TrainOptions(steps=5, batch=8, loss="rollout", horizon=3),
        seed=3,
    )
    assert np.isfinite(r2["final_loss"])


def test_windows_cover_every_start():
    traj = np.arange(2 * 6).reshape(2, 6, 1).astype(float)
    w = TR.windows(traj, 3)
    assert w.shape == (8, 3, 1)
    np.testing.assert_array_equal(w[0, :, 0], [0, 1, 2])


def test_valid_steps_and_blowup():
    curve = np.array([[0.0, 0.01, 0.05, 0.2, 0.3], [0.0, 0.01, 0.02, 0.03, 0.04]])
    np.testing.assert_array_equal(M.valid_steps(curve, 0.1), [2, 4])

    class Explode(torch.nn.Module):
        def forward(self, u):
            return u * 100.0

    out = M.rollout(Explode(), np.ones((1, 2)), 5, np.zeros(2), np.ones(2))
    assert np.isinf(out[0, -1]).all()


def test_conv_stepper_is_translation_equivariant_and_dense_is_not():
    torch.manual_seed(0)
    conv = ST.ConvStepper(PD.N_GRID, 0.05, 0.0, 1.0, 1.0)
    dense = ST.DenseStepper(PD.N_GRID, 0.05, 0.0, 1.0, 1.0)
    u = torch.as_tensor(np.sin(PD.grid()) + 0.3 * np.cos(3 * PD.grid()))[None]
    for m, equiv in ((conv, True), (dense, False)):
        with torch.no_grad():
            a = torch.roll(m(u), 5, -1)
            b = m(torch.roll(u, 5, -1))
        assert torch.allclose(a, b, atol=1e-12) is equiv
