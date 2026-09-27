"""The genetic-programming search and its operators."""

from __future__ import annotations

import itertools

import numpy as np

from physprior.symbolic.expressions import (
    OperatorSet,
    complexity,
    noise_floor,
    select,
    structure,
)
from physprior.symbolic.gp import (
    GPConfig,
    crossover,
    gp_search,
    point_mutation,
    random_tree,
)

OPS = OperatorSet()


def test_genetic_operators_produce_valid_trees():
    rng = np.random.default_rng(3)
    X = np.linspace(0.5, 2, 5)[:, None]
    from physprior.symbolic.expressions import evaluate

    for _ in range(50):
        a = random_tree(rng, OPS, 3)
        b = random_tree(rng, OPS, 3)
        for t in (crossover(rng, a, b), point_mutation(rng, a, OPS)):
            assert evaluate(t, X).shape == (5,)
    # point mutation keeps the shape of the tree
    a = random_tree(rng, OPS, 3, full=True)
    assert complexity(point_mutation(rng, a, OPS)) == complexity(a)


def test_gp_recovers_keplers_third_law_noise_free():
    a = np.geomspace(0.39, 5.2, 16)
    P = a**1.5
    r = gp_search(
        a[:, None],
        P,
        1 / P**2,
        ops=OPS,
        seed=3,
        config=GPConfig(population=80, generations=15),
        stop_loss=noise_floor(0.0, 16),
    )
    best = select(r.front)
    assert best.loss < 1e-20
    assert structure(best.tree) in {
        "sqrt(cube(x0))",
        "cube(sqrt(x0))",
        "*(x0,sqrt(x0))",
        "*(sqrt(x0),x0)",
    }
    assert r.reached_floor_at is not None
    # the recorded fronts are monotone: loss never rises between generations
    bests = [h["best_loss"] for h in r.history]
    assert all(b2 <= b1 for b1, b2 in itertools.pairwise(bests))
