"""The search space: exact counts, pruning, and exhaustive search."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from physprior.symbolic.enumerate import (
    count_by_depth,
    count_by_size,
    count_enumerated,
    exhaustive_search,
    iter_levels,
)
from physprior.symbolic.expressions import OperatorSet, depth, noise_floor, select


@pytest.mark.parametrize(
    "ops",
    [
        OperatorSet(("+", "*"), ()),
        OperatorSet(("+", "*"), ("square",)),
        OperatorSet(("+", "-", "*", "/"), ("exp", "log")),
    ],
)
def test_unpruned_enumeration_matches_the_recurrence(ops):
    got = count_enumerated(ops, 6, prune=False, max_consts=99)
    assert got == count_by_size(6, 2, len(ops.unary), len(ops.binary))


def test_depth_recurrence_matches_enumeration():
    ops = OperatorSet(("+", "*"), ("square",))
    levels = dict(iter_levels(ops, 7, prune=False, max_consts=99))
    for d, n in enumerate(count_by_depth(2, 2, 1, 2)):
        brute = sum(1 for ts in levels.values() for t in ts if depth(t) <= d)
        assert brute == n


def test_counts_grow_geometrically_and_pruning_removes_most():
    raw = count_by_size(10, 2, 5, 4)
    ratios = [b / a for a, b in itertools.pairwise(raw[3:])]
    assert min(ratios) > 5
    pr = count_enumerated(OperatorSet(), 6)
    assert pr[-1] < raw[5] / 2


def test_exhaustive_search_finds_the_inverse_square_law_and_stops():
    x = np.linspace(1.0, 4.0, 20)
    y = 3.0 / x**2
    r = exhaustive_search(
        x[:, None],
        y,
        1 / y**2,
        ops=OperatorSet(),
        max_size=6,
        stop_loss=noise_floor(0.0, 20),
    )
    assert r.reached_floor and r.size_reached == 4
    best = select(r.front)
    assert best.complexity == 4 and best.loss < 1e-20
