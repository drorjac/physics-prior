"""Genetic programming over weighting rules, with fitness measured by training.

The trees, operators and genetic operators are those of `physprior.symbolic`.
What differs from data-fitting SR is the fitness: a rule is scored by training
a PINN with it on every (task, seed) of the meta-training set and taking the
mean log10 error. There is no gradient to fit constants with, so constants are
searched by mutation on a log scale.

Every training run is a `Job`. Runs are cached on disk under
`.cache/lossdisc/`, keyed on the trial, the rule's full expression (constants
included), the task, the seed and the training configuration, so a search can
be resumed and a rule is never trained twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from physprior.config import get_settings
from physprior.symbolic.expressions import (
    Node,
    OperatorSet,
    bi,
    complexity,
    const,
    constants,
    to_string,
    with_constants,
)
from physprior.symbolic.gp import (
    crossover,
    point_mutation,
    random_tree,
    subtree_mutation,
)

from .pinn import ERR_CLIP, TrainConfig
from .rules import BalanceRule, ResidualRule
from .tasks import Inverse, Task

log = logging.getLogger(__name__)

OPS = OperatorSet(binary=("+", "-", "*", "/"), unary=("neg", "square", "exp"))


@dataclass(frozen=True)
class SearchConfig:
    population: int = 32
    generations: int = 12
    tournament: int = 3
    elite: int = 4
    p_crossover: float = 0.35
    p_subtree: float = 0.25
    p_point: float = 0.15
    p_scale: float = 0.10  # the rest: constant mutation
    max_size: int = 11
    init_depth: int = 3
    parsimony: float = 0.01  # decades of error per node
    seed: int = 0


# ---------------------------------------------------------------------------
# jobs and their cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    trial: str  # "forward" | "inverse"
    tree: Node
    task: Task | Inverse
    seed: int
    cfg: TrainConfig
    normalise: bool = True  # forward only

    def key(self) -> str:
        desc = {
            "trial": self.trial,
            "rule": to_string(self.tree, digits=17),
            "normalise": self.normalise,
            "task": repr(self.task),
            "seed": self.seed,
            "cfg": asdict(self.cfg),
        }
        return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()[
            :24
        ]


def _cache_dir() -> Path:
    p = get_settings().cache_dir / "lossdisc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _execute(job: Job) -> dict:
    import torch

    from .pinn import train_forward, train_inverse

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    if job.trial == "forward":
        assert isinstance(job.task, Task)
        out = train_forward(
            job.task, ResidualRule(job.tree, job.normalise), job.seed, job.cfg
        )
    else:
        assert isinstance(job.task, Inverse)
        out = train_inverse(job.task, BalanceRule(job.tree), job.seed, job.cfg)
    out["seconds"] = time.perf_counter() - t0
    return out


def _execute_cached(job: Job) -> dict:
    p = _cache_dir() / f"{job.key()}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = _execute(job)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(p)
    return out


class Runner:
    """Runs jobs in a process pool (or in this process with workers=1).

    A worker that dies (a segfault in a native library, the OS killing it)
    breaks the pool. The pool is then rebuilt and the batch resubmitted;
    finished runs come back from the cache, so only the lost ones rerun."""

    RETRIES = 10

    def __init__(self, workers: int = 1):
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def _start(self) -> None:
        # workers must not start Julia, and each trains on one thread
        os.environ.setdefault("PHYSPRIOR_NO_JULIA", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        # Accelerate's own threads: with them on, dgemm segfaults now and
        # then on macOS 13 when several processes train at once
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        self._pool = ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"))

    def __enter__(self) -> Runner:
        if self.workers > 1:
            self._start()
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def map(self, jobs: Sequence[Job]) -> list[dict]:
        if self._pool is None:
            return [_execute_cached(j) for j in jobs]
        for attempt in range(self.RETRIES + 1):
            try:
                return list(self._pool.map(_execute_cached, jobs))
            except BrokenProcessPool:
                if attempt == self.RETRIES:
                    raise
                log.warning(
                    "process pool broke; restarting it (attempt %d)", attempt + 1
                )
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._start()
        raise AssertionError("unreachable")


def score(trial: str, results: Sequence[dict]) -> float:
    """Mean log10 error over runs: the error itself for forward problems,
    the mean of log10 err_k and log10 err_u for inverse ones."""
    vals = []
    for r in results:
        if trial == "forward":
            vals.append(math.log10(max(r["err"], 1e-12)))
        else:
            ek = math.log10(max(r["err_k"], 1e-12))
            eu = math.log10(max(r["err_u"], 1e-12))
            vals.append(0.5 * (ek + eu))
    return float(np.mean(vals)) if vals else math.log10(ERR_CLIP)


def evaluate_rules(
    runner: Runner,
    trial: str,
    trees: Sequence[Node],
    tasks: Sequence[Task | Inverse],
    seeds: Sequence[int],
    cfg: TrainConfig,
    normalise: bool = True,
) -> list[tuple[float, list[dict]]]:
    """Score every tree on every (task, seed), in one batch of jobs."""
    jobs = [
        Job(trial, t, task, s, cfg, normalise)
        for t in trees
        for task in tasks
        for s in seeds
    ]
    res = runner.map(jobs)
    per = len(tasks) * len(seeds)
    out = []
    for i in range(len(trees)):
        chunk = res[i * per : (i + 1) * per]
        out.append((score(trial, chunk), chunk))
    return out


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def _random_constants(rng: np.random.Generator, t: Node) -> Node:
    """Leaf constants log-uniform in [0.1, 30], either sign."""
    n = len(constants(t))
    if n == 0:
        return t
    mag = np.exp(rng.uniform(math.log(0.1), math.log(30.0), n))
    return with_constants(t, mag * rng.choice([-1.0, 1.0], n))


def _mutate_constants(rng: np.random.Generator, t: Node) -> Node:
    c = np.asarray(constants(t), float)
    if c.size == 0:
        return t
    i = int(rng.integers(c.size))
    c = c.copy()
    c[i] *= math.exp(0.7 * rng.standard_normal())
    if rng.random() < 0.1:
        c[i] = -c[i]
    return with_constants(t, c)


def _scale(rng: np.random.Generator, t: Node) -> Node:
    """c * t: the strength of a weighting is a constant the trees otherwise
    reach only by chance."""
    c = math.exp(rng.uniform(math.log(0.3), math.log(30.0)))
    return bi("*", const(c), t)


@dataclass
class Evaluated:
    tree: Node
    fitness: float  # mean log10 error
    size: int
    label: str

    def cost(self, parsimony: float) -> float:
        return self.fitness + parsimony * self.size


@dataclass
class SearchResult:
    evaluated: list[Evaluated]
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def best(self, k: int, parsimony: float) -> list[Evaluated]:
        return sorted(self.evaluated, key=lambda e: e.cost(parsimony))[:k]


def search(
    fitness: Callable[[list[Node]], list[float]],
    n_features: int,
    seeds_init: Sequence[Node] = (),
    config: SearchConfig = SearchConfig(),
    names: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> SearchResult:
    """Steady GP with elitism. `fitness` scores a batch of trees (lower is
    better); it is called once per generation on the trees not yet seen."""
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    seen: dict[str, Evaluated] = {}
    t0 = time.perf_counter()

    def label(t: Node) -> str:
        return to_string(t, list(names) if names else None, digits=4)

    def score_batch(trees: list[Node]) -> list[Evaluated]:
        fresh: dict[str, Node] = {}
        for t in trees:
            key = to_string(t, digits=17)
            if key not in seen and key not in fresh:
                fresh[key] = t
        if fresh:
            vals = fitness(list(fresh.values()))
            for (key, t), v in zip(fresh.items(), vals, strict=True):
                v = v if math.isfinite(v) else math.log10(ERR_CLIP)
                seen[key] = Evaluated(t, v, complexity(t), label(t))
        return [seen[to_string(t, digits=17)] for t in trees]

    init: list[Node] = list(seeds_init)
    while len(init) < cfg.population:
        depth = 1 + len(init) % cfg.init_depth
        t = random_tree(rng, OPS, depth, n_features, full=bool(len(init) % 2))
        if complexity(t) <= cfg.max_size:
            init.append(_random_constants(rng, t))
    pop = score_batch(init)

    def tournament() -> Evaluated:
        idx = rng.integers(len(pop), size=cfg.tournament)
        return min((pop[int(i)] for i in idx), key=lambda e: e.cost(cfg.parsimony))

    probs = np.cumsum([cfg.p_crossover, cfg.p_subtree, cfg.p_point, cfg.p_scale])
    history: list[dict] = []
    for gen in range(cfg.generations):
        elite = sorted(pop, key=lambda e: e.cost(cfg.parsimony))[: cfg.elite]
        children: list[Node] = []
        keys: set[str] = set()
        tries = 0
        while len(children) < cfg.population - cfg.elite:
            tries += 1
            parent = tournament().tree
            r = rng.random()
            if r < probs[0]:
                child = crossover(rng, parent, tournament().tree)
            elif r < probs[1]:
                child = _random_constants(
                    rng, subtree_mutation(rng, parent, OPS, n_features)
                )
            elif r < probs[2]:
                child = point_mutation(rng, parent, OPS, n_features)
            elif r < probs[3]:
                child = _scale(rng, parent)
            else:
                child = _mutate_constants(rng, parent)
            if complexity(child) > cfg.max_size:
                continue
            # a child already trained costs a slot and teaches nothing, so
            # duplicates are redrawn (up to a limit, for a converged population)
            key = to_string(child, digits=17)
            if (key in seen or key in keys) and tries < 50 * cfg.population:
                continue
            keys.add(key)
            children.append(child)
        pop = elite + score_batch(children)
        best = min(seen.values(), key=lambda e: e.cost(cfg.parsimony))
        row = {
            "generation": gen,
            "best_fitness": best.fitness,
            "best_size": best.size,
            "best_rule": best.label,
            "n_evaluated": len(seen),
            "seconds": time.perf_counter() - t0,
        }
        if log:
            log(f"gen {gen}: best {best.fitness:.3f} ({best.label}), {len(seen)} rules")
        history.append(row)

    return SearchResult(list(seen.values()), history, time.perf_counter() - t0)
