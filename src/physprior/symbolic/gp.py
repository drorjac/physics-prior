"""A small genetic-programming symbolic regressor.

The textbook algorithm (Koza 1992) with the three additions that make modern
SR tools work on physics data, each of which PySR also has:

1. Constants are FITTED, not evolved. Every new structure has its constants
   optimised (Levenberg-Marquardt or BFGS) once, and the result is cached by
   structure. Evolution then only has to find the form.
2. A hall of fame keeps the best expression at every complexity: the Pareto
   front of loss against size. It is re-injected into each generation
   (elitism), so a good simple law is not lost while the population explores
   larger trees.
3. The answer is chosen off the front by PySR's score,
   -d log(loss) / d complexity (`expressions.select`), not by lowest loss.

The loop, per generation:

    parents   tournament selection on cost = loss/baseline + parsimony*size
    children  subtree crossover | subtree mutation | point mutation |
              constant perturbation | copy
    evaluate  fit constants (cached per structure), update the front
    replace   generational: front + children form the next population

Deliberately simple and pure Python/numpy. It is here to be read and
measured against exhaustive search and PySR, not to compete with PySR on
speed. Differences from PySR that matter are listed in
docs/theory/symbolic_regression.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .expressions import (
    Candidate,
    Node,
    OperatorSet,
    bi,
    complexity,
    const,
    constants,
    fit_constants,
    get,
    loss,
    pareto_front,
    paths,
    put,
    structure,
    un,
    var,
    with_constants,
)


@dataclass(frozen=True)
class GPConfig:
    population: int = 150
    generations: int = 40
    tournament: int = 5
    p_crossover: float = 0.40
    p_subtree: float = 0.20
    p_point: float = 0.25
    p_constant: float = 0.05  # the rest are copies
    max_size: int = 15
    init_depth: int = 3
    parsimony: float = 1e-3
    optimizer: str = "lm"  # "lm" | "bfgs"
    p_optimize: float = 1.0  # chance a NEW structure has its constants fitted
    patience: int = 5  # generations run after the noise floor is reached
    elitism: bool = True
    p_leaf_var: float = 0.6


@dataclass
class GPResult:
    front: list[Candidate]
    history: list[dict]
    fronts: list[list[tuple[int, float, str]]]  # the front after each generation
    n_structures: int  # distinct forms whose constants were fitted
    n_evaluated: int
    seconds: float
    reached_floor_at: int | None = None  # generation, or None
    config: GPConfig = field(default_factory=GPConfig)


# ---------------------------------------------------------------------------
# random trees and the genetic operators
# ---------------------------------------------------------------------------


def random_tree(
    rng: np.random.Generator,
    ops: OperatorSet,
    depth: int,
    n_features: int = 1,
    p_leaf_var: float = 0.6,
    full: bool = False,
) -> Node:
    """'grow' (leaves may appear early) or 'full' (all branches to `depth`)."""
    n_u, n_b = len(ops.unary), len(ops.binary)
    if depth == 0 or (not full and rng.random() < 0.3):
        if rng.random() < p_leaf_var:
            return var(int(rng.integers(n_features)))
        return const(1.0)
    k = int(rng.integers(n_u + n_b))
    if k < n_u:
        return un(
            ops.unary[k], random_tree(rng, ops, depth - 1, n_features, p_leaf_var, full)
        )
    op = ops.binary[k - n_u]
    return bi(
        op,
        random_tree(rng, ops, depth - 1, n_features, p_leaf_var, full),
        random_tree(rng, ops, depth - 1, n_features, p_leaf_var, full),
    )


def crossover(rng: np.random.Generator, a: Node, b: Node) -> Node:
    """Subtree crossover: a random subtree of `a` replaced by one of `b`."""
    pa = list(paths(a))
    pb = list(paths(b))
    return put(
        a, pa[int(rng.integers(len(pa)))], get(b, pb[int(rng.integers(len(pb)))])
    )


def subtree_mutation(
    rng: np.random.Generator, t: Node, ops: OperatorSet, n_features: int = 1
) -> Node:
    ps = list(paths(t))
    return put(t, ps[int(rng.integers(len(ps)))], random_tree(rng, ops, 2, n_features))


def point_mutation(
    rng: np.random.Generator, t: Node, ops: OperatorSet, n_features: int = 1
) -> Node:
    """Change one node's label, keeping its arity."""
    ps = list(paths(t))
    p = ps[int(rng.integers(len(ps)))]
    n = get(t, p)
    if n.op == "x":
        new = (
            const(1.0)
            if n_features == 1 or rng.random() < 0.5
            else var(int(rng.integers(n_features)))
        )
    elif n.op == "c":
        new = var(int(rng.integers(n_features)))
    elif len(n.children) == 1:
        new = Node(str(rng.choice(ops.unary)), n.children) if ops.unary else n
    else:
        new = Node(str(rng.choice(ops.binary)), n.children)
    return put(t, p, new)


def perturb_constants(rng: np.random.Generator, t: Node, scale: float = 0.3) -> Node:
    c = np.asarray(constants(t), float)
    if c.size == 0:
        return t
    return with_constants(t, c * (1.0 + scale * rng.standard_normal(c.size)))


def _tournament(rng, pop: list[Candidate], cost: np.ndarray, k: int) -> Candidate:
    idx = rng.integers(len(pop), size=k)
    return pop[int(idx[np.argmin(cost[idx])])]


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def gp_search(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    *,
    ops: OperatorSet,
    config: GPConfig | None = None,
    seed: int = 0,
    stop_loss: float | None = None,
) -> GPResult:
    """Run the GP. `stop_loss` is the measurement-noise floor: once the front
    reaches it the search runs `patience` more generations (so simpler forms
    at the same loss can appear) and stops."""
    cfg = config or GPConfig()
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    d = X.shape[1]
    t0 = time.perf_counter()

    # loss of the best constant model, so cost is scale-free
    ww = np.ones_like(y) if w is None else w
    c_best = float(np.sum(ww * y) / np.sum(ww))
    baseline = max(float(np.mean(ww * (y - c_best) ** 2)), 1e-300)

    cache: dict[str, tuple[Node, float]] = {}
    n_eval = 0

    def evaluate(t: Node) -> Candidate:
        nonlocal n_eval
        n_eval += 1
        key = structure(t)
        hit = cache.get(key)
        if hit is not None:
            return Candidate.of(hit[0], hit[1])
        if rng.random() < cfg.p_optimize:
            ft, val = fit_constants(
                t,
                X,
                y,
                w,
                method=cfg.optimizer,
                restarts=1,
                rng=rng,
                good_enough=stop_loss or 0.0,
            )
            cache[key] = (ft, val)
            return Candidate.of(ft, val)
        return Candidate.of(t, loss(t, X, y, w))

    # ramped half-and-half initialisation
    pop: list[Candidate] = []
    for i in range(cfg.population):
        depth = 1 + i % cfg.init_depth
        t = random_tree(rng, ops, depth, d, cfg.p_leaf_var, full=bool(i % 2))
        if complexity(t) <= cfg.max_size:
            pop.append(evaluate(t))
    hof = pareto_front(pop)

    history: list[dict] = []
    fronts: list[list[tuple[int, float, str]]] = []
    reached_at: int | None = None
    probs = np.array(
        [cfg.p_crossover, cfg.p_subtree, cfg.p_point, cfg.p_constant], float
    )
    for gen in range(cfg.generations):
        cost = np.array([c.loss / baseline + cfg.parsimony * c.complexity for c in pop])
        cost = np.where(np.isfinite(cost), cost, np.inf)
        children: list[Candidate] = list(hof) if cfg.elitism else []
        while len(children) < cfg.population:
            parent = _tournament(rng, pop, cost, cfg.tournament)
            r = rng.random()
            child: Node | None = None
            for _ in range(3):
                if r < probs[0]:
                    other = _tournament(rng, pop, cost, cfg.tournament)
                    child = crossover(rng, parent.tree, other.tree)
                elif r < probs[:2].sum():
                    child = subtree_mutation(rng, parent.tree, ops, d)
                elif r < probs[:3].sum():
                    child = point_mutation(rng, parent.tree, ops, d)
                elif r < probs.sum():
                    child = perturb_constants(rng, parent.tree)
                else:
                    child = parent.tree
                if complexity(child) <= cfg.max_size:
                    break
                child = None
            if child is None or child is parent.tree:
                children.append(parent)
                continue
            children.append(evaluate(child))
        pop = children
        hof = pareto_front(list(hof) + pop)
        best = min(c.loss for c in hof)
        history.append(
            {
                "generation": gen,
                "best_loss": best,
                "front_size": len(hof),
                "n_structures": len(cache),
                "n_evaluated": n_eval,
                "seconds": time.perf_counter() - t0,
            }
        )
        fronts.append([(c.complexity, c.loss, c.expression()) for c in hof])
        if stop_loss is not None and best <= stop_loss and reached_at is None:
            reached_at = gen
        if reached_at is not None and gen - reached_at >= cfg.patience:
            break

    return GPResult(
        front=hof,
        history=history,
        fronts=fronts,
        n_structures=len(cache),
        n_evaluated=n_eval,
        seconds=time.perf_counter() - t0,
        reached_floor_at=reached_at,
        config=cfg,
    )
