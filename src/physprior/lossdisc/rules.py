"""Weighting rules as expression trees, and the hand-designed baselines.

A rule is a `physprior.symbolic` tree. It is evaluated on detached training
state, so it shapes the loss without being differentiated.

ResidualRule (trial A), features in this order:

    x0  t     collocation time in [0, 1]
    x1  C     sum_{j<i} r_j^2 / N, cumulative residual at earlier times
    x2  q     the same sum as a fraction of the total, in [0, 1]
    x3  tau   training progress, step / steps

    w_i = exp(g_i) / mean_j exp(g_j)       (normalised)
    w_i = exp(g_i)                         (normalise=False, as published)

BalanceRule (trial B), features in this order:

    x0  tau
    x1  rho     log10(L_data / L_phys)
    x2  gamma   log10(|grad L_data| / |grad L_phys|), network weights only

    lambda = exp(h)

The baselines are trees in the same space: the plain PINN is g = 0, causal
weighting is g = -eps C, a fixed weight is h = ln lambda, gradient-norm
balancing is h = ln(10) gamma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from physprior.symbolic.expressions import Node, bi, const, evaluate, to_string, var

RESIDUAL_FEATURES = ("t", "C", "q", "tau")
BALANCE_FEATURES = ("tau", "rho", "gamma")
EXP_CLIP = 50.0


class InvalidRule(ValueError):
    """The rule returned NaN or inf on the current training state."""


@dataclass(frozen=True)
class ResidualRule:
    tree: Node
    normalise: bool = True

    def features(self, x: np.ndarray, r2: np.ndarray, tau: float) -> np.ndarray:
        n = len(r2)
        csum = np.cumsum(r2)
        before = (csum - r2) / n
        total = csum[-1] / n
        q = before / total if total > 0 else np.zeros(n)
        return np.column_stack([x, before, q, np.full(n, tau)])

    def weights(self, x: np.ndarray, r2: np.ndarray, tau: float) -> np.ndarray:
        g = evaluate(self.tree, self.features(x, r2, tau))
        if not np.all(np.isfinite(g)):
            raise InvalidRule(to_string(self.tree))
        w = np.exp(np.clip(g, -EXP_CLIP, EXP_CLIP))
        if self.normalise:
            w = w / w.mean()
        return w

    def label(self) -> str:
        return to_string(self.tree, list(RESIDUAL_FEATURES), digits=4)


@dataclass(frozen=True)
class BalanceRule:
    tree: Node

    @property
    def needs_gradients(self) -> bool:
        return _uses(self.tree, 2)

    def weight(self, tau: float, rho: float, gamma: float) -> float:
        h = float(evaluate(self.tree, np.array([[tau, rho, gamma]]))[0])
        if not math.isfinite(h):
            raise InvalidRule(to_string(self.tree))
        return math.exp(min(max(h, -EXP_CLIP), EXP_CLIP))

    def label(self) -> str:
        return to_string(self.tree, list(BALANCE_FEATURES), digits=4)


def to_json(t: Node) -> list:
    """A tree as nested lists, constants at full precision."""
    if t.op == "x":
        return ["x", t.feature]
    if t.op == "c":
        return ["c", t.value]
    return [t.op, *(to_json(c) for c in t.children)]


def from_json(a: list) -> Node:
    if a[0] == "x":
        return var(int(a[1]))
    if a[0] == "c":
        return const(float(a[1]))
    return Node(a[0], tuple(from_json(c) for c in a[1:]))


def _uses(t: Node, feature: int) -> bool:
    if t.op == "x":
        return t.feature == feature
    return any(_uses(c, feature) for c in t.children)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def uniform() -> ResidualRule:
    return ResidualRule(const(0.0))


def causal(eps: float, normalise: bool = False) -> ResidualRule:
    """Wang, Sankaran & Perdikaris (2024): w_i = exp(-eps sum_{j<i} L_j)."""
    return ResidualRule(bi("*", const(-eps), var(1)), normalise=normalise)


def fixed(lam: float) -> BalanceRule:
    return BalanceRule(const(math.log(lam)))


def gradnorm(scale: float = 1.0) -> BalanceRule:
    """Wang, Teng & Perdikaris (2021) in its simplest form: lambda is the
    ratio of gradient norms, so both terms pull with the same force."""
    return BalanceRule(
        bi("+", const(math.log(scale)), bi("*", const(math.log(10)), var(2)))
    )
