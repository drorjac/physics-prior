"""The from-scratch network: its gradient, and its hand-written optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.optim import scratch as S

torch = pytest.importorskip("torch")


def _net_and_data(act="tanh", seed=1):
    net = S.init_mlp([1, 8, 8, 1], act, seed=seed)
    x, y, _ = S.demo_data(20)
    return net, x, y


@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_backprop_matches_finite_differences(act):
    net, x, y = _net_and_data(act)
    _, g = S.loss_and_grad(net, x, y)
    gfd = S.finite_difference_grad(net, x, y, h=1e-5)
    assert np.linalg.norm(gfd - g) / np.linalg.norm(g) < 1e-7


@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_backprop_matches_torch_autograd(act):
    net, x, y = _net_and_data(act)
    _, g = S.loss_and_grad(net, x, y)
    assert np.linalg.norm(S.torch_grad(net, x, y) - g) / np.linalg.norm(g) < 1e-12


def test_finite_difference_check_converges_at_second_order():
    net, x, y = _net_and_data()
    rows = S.fd_convergence(net, x, y, steps=(1e-2, 1e-3))
    slope = np.log10(rows[0]["rel_error"] / rows[1]["rel_error"])
    assert 1.8 < slope < 2.2


def test_flat_round_trip():
    net, _, _ = _net_and_data()
    theta = net.flat()
    net.set_flat(theta * 2.0)
    assert np.allclose(net.flat(), 2.0 * theta)
    assert theta.size == net.n_params()


def test_init_scale_matches_the_scheme():
    net = S.init_mlp([400, 400, 400], "tanh", "xavier", seed=0)
    assert net.W[0].std() == pytest.approx(np.sqrt(2.0 / 800), rel=0.02)
    net = S.init_mlp([400, 400, 400], "relu", seed=0)  # he by default
    assert net.W[0].std() == pytest.approx(np.sqrt(2.0 / 400), rel=0.02)


def test_adam_first_step_is_lr_with_bias_correction():
    theta = np.zeros(3)
    g = np.array([1e-3, 1.0, -50.0])
    step = S.Adam(lr=0.01).step(theta, g)
    assert np.allclose(np.abs(step), 0.01, rtol=1e-4)
    raw = S.Adam(lr=0.01, bias_correction=False).step(theta, g)
    # (1 - beta1) / sqrt(1 - beta2) = 0.1 / sqrt(0.001)
    assert np.allclose(np.abs(raw), 0.01 * 0.1 / np.sqrt(0.001), rtol=1e-3)


def _torch_trajectory(opt_factory, A, theta0, steps):
    p = torch.tensor(theta0, requires_grad=True)
    opt = opt_factory([p])
    At = torch.tensor(A)
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * p @ At @ p).backward()
        opt.step()
    return p.detach().numpy()


@pytest.mark.parametrize(
    "name,factory",
    [
        ("sgd", lambda ps: torch.optim.SGD(ps, lr=0.05)),
        ("momentum", lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
        (
            "nesterov",
            lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, nesterov=True),
        ),
        ("rmsprop", lambda ps: torch.optim.RMSprop(ps, lr=0.05, alpha=0.99, eps=1e-8)),
        ("adam", lambda ps: torch.optim.Adam(ps, lr=0.05)),
    ],
)
def test_hand_written_optimizers_match_torch(name, factory):
    """Same quadratic, same start: the hand-written update and torch's agree."""
    A = np.diag([1.0, 10.0])
    theta0 = np.array([1.0, -1.0])
    ours = S.make_optimizer(name, 0.05)
    th = theta0.copy()
    for _ in range(25):
        th = ours.step(th, A @ th)
    assert np.allclose(th, _torch_trajectory(factory, A, theta0, 25), atol=1e-10)


def test_training_lowers_the_loss():
    net = S.init_mlp([1, 16, 16, 1], "tanh", seed=0)
    x, y, _ = S.demo_data(32)
    hist = S.train(net, x, y, S.Adam(lr=3e-3), steps=300, record_every=50)
    assert hist["loss"][-1] < 0.2 * hist["loss"][0]
