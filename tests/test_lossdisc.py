"""The learned-loss-weighting study: derivatives, rules, search, one tiny run."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.lossdisc import rules as R  # noqa: E402
from physprior.lossdisc.pinn import (  # noqa: E402
    ERR_CLIP,
    Net,
    TrainConfig,
    train_forward,
    train_inverse,
)
from physprior.lossdisc.search import SearchConfig, score, search  # noqa: E402
from physprior.lossdisc.tasks import (  # noqa: E402
    FORWARD_TEST,
    FORWARD_TRAIN,
    INVERSE_TEST,
    INVERSE_TRAIN,
    Inverse,
    Task,
)
from physprior.symbolic.expressions import bi, const, un, var  # noqa: E402

TINY = TrainConfig(width=8, depth=2, steps=20, n_col=16)


def test_forward_mode_derivatives_match_autograd():
    net = Net(16, 3, seed=0)
    s = torch.linspace(-1, 1, 11, dtype=torch.float64)[:, None].requires_grad_(True)
    u, du, ddu = net(s)
    (g,) = torch.autograd.grad(u.sum(), s, create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), s)
    assert torch.allclose(du, g, atol=1e-12)
    assert torch.allclose(ddu, gg, atol=1e-12)


def test_reference_solution_of_the_oscillator():
    task = Task("sho", "sho", 2.0)
    t, u = task.solution()
    assert np.max(np.abs(u - np.cos(2 * np.pi * 2.0 * t))) < 1e-8


def test_task_sets_are_disjoint():
    for a, b in ((FORWARD_TRAIN, FORWARD_TEST), (INVERSE_TRAIN, INVERSE_TEST)):
        assert not {x.name for x in a} & {x.name for x in b}


def test_inverse_noise_depends_on_the_seed():
    inv = Inverse(Task("sho", "sho", 2.0))
    _, y3 = inv.observations(3)
    _, y11 = inv.observations(11)
    assert not np.allclose(y3, y11)
    np.testing.assert_array_equal(y3, inv.observations(3)[1])


def test_uniform_and_causal_are_points_of_the_rule_space():
    x = (np.arange(8) + 0.5) / 8
    r2 = np.linspace(1.0, 2.0, 8)
    np.testing.assert_allclose(R.uniform().weights(x, r2, 0.3), 1.0)
    eps = 5.0
    before = (np.cumsum(r2) - r2) / len(r2)
    np.testing.assert_allclose(R.causal(eps).weights(x, r2, 0.3), np.exp(-eps * before))
    w = R.ResidualRule(R.causal(eps).tree, normalise=True).weights(x, r2, 0.3)
    assert w.mean() == pytest.approx(1.0)


def test_balance_baselines():
    assert R.fixed(0.1).weight(0.5, 1.0, 2.0) == pytest.approx(0.1)
    assert R.gradnorm(2.0).weight(0.5, 1.0, 1.5) == pytest.approx(2.0 * 10**1.5)
    assert R.gradnorm().needs_gradients
    assert not R.fixed(1.0).needs_gradients


def test_invalid_rule_is_reported():
    bad = R.ResidualRule(un("log", un("neg", bi("+", var(0), const(1.0)))))
    with pytest.raises(R.InvalidRule):
        bad.weights(np.array([0.1, 0.2]), np.array([1.0, 1.0]), 0.0)
    out = train_forward(FORWARD_TRAIN[0], bad, 3, TINY)
    assert out["valid"] is False
    assert out["err"] == ERR_CLIP


def test_json_round_trip():
    t = bi("*", const(-math.pi), bi("+", var(1), un("square", var(3))))
    assert R.from_json(R.to_json(t)) == t


def test_tiny_training_runs():
    f = train_forward(FORWARD_TRAIN[0], R.causal(10.0), 3, TINY)
    assert f["valid"] and 0 < f["err"] <= ERR_CLIP
    i = train_inverse(INVERSE_TRAIN[0], R.gradnorm(), 3, TINY)
    assert i["valid"] and np.isfinite(i["k_hat"])


def test_score_is_mean_log10():
    assert score("forward", [{"err": 0.1}, {"err": 0.001}]) == pytest.approx(-2.0)
    s = score("inverse", [{"err_k": 0.01, "err_u": 0.1}])
    assert s == pytest.approx(-1.5)


def test_search_finds_a_known_minimum():
    # a cheap stand-in for training: distance of the rule at three points
    # from -3 x1, the causal form with eps = 3
    X = np.array([[0.1, 0.5, 0.2, 0.0], [0.5, 1.0, 0.6, 0.5], [0.9, 2.0, 0.9, 1.0]])

    def fitness(trees):
        from physprior.symbolic.expressions import evaluate

        out = []
        for t in trees:
            g = evaluate(t, X)
            d = float(np.mean((g + 3 * X[:, 1]) ** 2))
            out.append(math.log10(d + 1e-6) if np.isfinite(d) else 1.0)
        return out

    res = search(
        fitness,
        4,
        seeds_init=[const(0.0), bi("*", const(-2.0), var(1))],
        config=SearchConfig(population=16, generations=6, seed=1),
    )
    best = res.best(1, 0.0)[0]
    assert best.fitness < fitness([const(0.0)])[0]
    assert len(res.history) == 6
