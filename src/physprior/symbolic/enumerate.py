"""Exhaustive search: every tree up to a size, each with its constants fitted.

This is the baseline every heuristic search is an approximation to, and the
reason heuristics are needed at all. With L leaf types, U unary and B binary
operators, the number of distinct trees of size s obeys

    T(1) = L
    T(s) = U T(s-1) + B sum_{i+j=s-1} T(i) T(j)

which grows geometrically in s (the binary term is a Catalan-type
convolution). `count_by_size` and `count_by_depth` evaluate the recurrences
exactly; `enumerate_trees` generates the trees themselves, with the syntactic
redundancies removed (see `_keep`), so the two can be compared.

The search walks sizes in increasing order and stops at the first size whose
best fit reaches the measurement-noise floor -- the searcher knows the error
bars, not the law.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .expressions import (
    COMMUTATIVE,
    Candidate,
    Node,
    OperatorSet,
    bi,
    const,
    fit_constants,
    loss,
    pareto_front,
    un,
    var,
)

# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------


def count_by_size(
    max_size: int, n_leaves: int, n_unary: int, n_binary: int
) -> list[int]:
    """Number of distinct labelled trees of each size 1..max_size (no pruning).
    Python ints: these overflow int64 quickly."""
    T = [0] * (max_size + 1)
    for s in range(1, max_size + 1):
        if s == 1:
            T[s] = n_leaves
            continue
        conv = sum(T[i] * T[s - 1 - i] for i in range(1, s - 1))
        T[s] = n_unary * T[s - 1] + n_binary * conv
    return T[1:]


def count_by_depth(
    max_depth: int, n_leaves: int, n_unary: int, n_binary: int
) -> list[int]:
    """Number of trees of depth <= d for d = 0..max_depth: D(d) = L + U D(d-1)
    + B D(d-1)^2. Doubly exponential in depth."""
    D = [n_leaves]
    for _ in range(max_depth):
        prev = D[-1]
        D.append(n_leaves + n_unary * prev + n_binary * prev * prev)
    return D


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

# pairs that undo each other; applying one to the other is a longer spelling
# of a tree that already exists
_INVERSES = {
    ("exp", "log"),
    ("log", "exp"),
    ("square", "sqrt"),
    ("sqrt", "square"),
    ("neg", "neg"),
    ("inv", "inv"),
}


@dataclass(frozen=True)
class _Item:
    node: Node
    key: str  # structure string, for canonical ordering
    n_const: int
    has_var: bool


def _leaf_items(n_features: int) -> list[_Item]:
    out = [_Item(var(i), f"x{i}", 0, True) for i in range(n_features)]
    out.append(_Item(const(1.0), "c", 1, False))
    return out


def iter_levels(
    ops: OperatorSet,
    max_size: int,
    n_features: int = 1,
    max_consts: int = 2,
    prune: bool = True,
) -> Iterator[tuple[int, list[Node]]]:
    """Yield (size, trees of that size), smallest first, building each level
    from the smaller ones only when it is asked for. With `prune`, drop
    syntactic redundancy:

    - an operator applied only to constants (it is just a constant);
    - a commutative operator with its operands out of canonical order;
    - x - x and x / x with identical operands;
    - an operator applied to its own inverse (exp(log(.)), sqrt(square(.)));
    - more than `max_consts` free constants;
    - pow with anything but a leaf as exponent (the constraint the repo's
      wrapper gives PySR).

    Without `prune`, level s has exactly `count_by_size(...)[s-1]` trees,
    which the tests check.
    """
    by: dict[int, list[_Item]] = {1: _leaf_items(n_features)}
    yield 1, [it.node for it in by[1] if it.has_var or not prune]
    for s in range(2, max_size + 1):
        items: list[_Item] = []
        for op in ops.unary:
            for a in by[s - 1]:
                if prune and (not a.has_var or (op, a.node.op) in _INVERSES):
                    continue
                items.append(
                    _Item(un(op, a.node), f"{op}({a.key})", a.n_const, a.has_var)
                )
        for op in ops.binary:
            for i in range(1, s - 1):
                j = s - 1 - i
                for a in by[i]:
                    for b in by[j]:
                        nc = a.n_const + b.n_const
                        hv = a.has_var or b.has_var
                        if prune:
                            if not hv or nc > max_consts:
                                continue
                            if op in COMMUTATIVE and a.key > b.key:
                                continue
                            if op in ("-", "/") and a.key == b.key and a.has_var:
                                continue
                            if op == "pow" and j != 1:
                                continue
                        items.append(
                            _Item(
                                bi(op, a.node, b.node), f"{op}({a.key},{b.key})", nc, hv
                            )
                        )
        by[s] = items
        yield s, [it.node for it in items if it.has_var or not prune]


def enumerate_trees(
    ops: OperatorSet,
    max_size: int,
    n_features: int = 1,
    max_consts: int = 2,
    prune: bool = True,
) -> dict[int, list[Node]]:
    """All trees by size; see `iter_levels` for what pruning removes."""
    return dict(iter_levels(ops, max_size, n_features, max_consts, prune))


def count_enumerated(ops: OperatorSet, max_size: int, **kw) -> list[int]:
    trees = enumerate_trees(ops, max_size, **kw)
    return [len(trees[s]) for s in range(1, max_size + 1)]


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


@dataclass
class ExhaustiveResult:
    front: list[Candidate]
    n_evaluated: int
    seconds: float
    size_reached: int
    reached_floor: bool
    history: list[dict] = field(default_factory=list)  # one row per size
    timed_out: bool = False


def exhaustive_search(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    *,
    ops: OperatorSet,
    max_size: int = 7,
    max_consts: int = 2,
    stop_loss: float | None = None,
    time_budget: float | None = None,
    method: str = "lm",
    seed: int = 0,
) -> ExhaustiveResult:
    """Fit every tree up to `max_size`, smallest first.

    Stops after the first size at which the best loss is <= `stop_loss`
    (the whole size is finished, so the front at that size is complete), or
    when `time_budget` seconds have passed.
    """
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    cands: list[Candidate] = []
    n_eval = 0
    history: list[dict] = []
    reached = False
    timed_out = False
    size = 0
    t_enum = 0.0
    levels = iter_levels(ops, max_size, n_features=X.shape[1], max_consts=max_consts)
    while True:
        te = time.perf_counter()
        try:
            size, level = next(levels)
        except StopIteration:
            break
        t_enum += time.perf_counter() - te
        best = float("inf")
        for t in level:
            ft, val = (
                fit_constants(
                    t,
                    X,
                    y,
                    w,
                    method=method,
                    restarts=1,
                    rng=rng,
                    good_enough=stop_loss or 0.0,
                )
                if _has_const(t)
                else (t, loss(t, X, y, w))
            )
            n_eval += 1
            if np.isfinite(val):
                cands.append(Candidate.of(ft, val))
                best = min(best, val)
            if time_budget is not None and time.perf_counter() - t0 > time_budget:
                timed_out = True
                break
        history.append(
            {
                "size": size,
                "n_trees": len(level),
                "enumeration_seconds_cum": t_enum,
                "n_evaluated_cum": n_eval,
                "seconds_cum": time.perf_counter() - t0,
                "best_loss": best,
            }
        )
        if timed_out:
            break
        if stop_loss is not None and best <= stop_loss:
            reached = True
            break
    return ExhaustiveResult(
        front=pareto_front(cands),
        n_evaluated=n_eval,
        seconds=time.perf_counter() - t0,
        size_reached=size,
        reached_floor=reached,
        history=history,
        timed_out=timed_out,
    )


def _has_const(t: Node) -> bool:
    return t.op == "c" or any(_has_const(c) for c in t.children)
