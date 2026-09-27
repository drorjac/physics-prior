"""Expression trees: the objects symbolic regression searches over.

A candidate law is a tree. Internal nodes are operators (unary or binary),
leaves are either an input feature or a free constant. Three things are
needed of a tree and all live here:

`evaluate`         numbers in, numbers out, with invalid arithmetic (log of a
                   negative, overflow in exp) mapped to NaN/inf so the search
                   can reject the candidate rather than crash;
`complexity`       the node count -- the same default measure PySR uses
                   (`count_nodes` in SymbolicRegression.jl), so the Pareto
                   fronts of the in-repo searches and of PySR share an axis;
`fit_constants`    the continuous half of the problem. A tree fixes the form;
                   its constants are then found by local optimisation
                   (Levenberg-Marquardt or BFGS), from several starts.

Trees are immutable. Constants live in the leaves, so a fitted tree is a new
tree with new leaf values.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "neg": np.negative,
    "sqrt": np.sqrt,  # NaN below zero, as PySR's default sqrt
    "square": np.square,
    "cube": lambda a: a * a * a,
    "exp": np.exp,
    "log": np.log,  # NaN below zero, -inf at zero
    "inv": lambda a: 1.0 / a,
    "sin": np.sin,
    "cos": np.cos,
}

BINARY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "pow": np.power,
}

COMMUTATIVE = frozenset({"+", "*"})

# sympy spelling of each operator, used by `to_string`
_UNARY_FMT = {
    "neg": "(-{0})",
    "sqrt": "sqrt({0})",
    "square": "({0})**2",
    "cube": "({0})**3",
    "exp": "exp({0})",
    "log": "log({0})",
    "inv": "(1/{0})",
    "sin": "sin({0})",
    "cos": "cos({0})",
}


@dataclass(frozen=True)
class OperatorSet:
    """The vocabulary a search may use. It is a prior: nothing outside it can
    be written down, whatever the data say."""

    binary: tuple[str, ...] = ("+", "-", "*", "/")
    unary: tuple[str, ...] = ("sqrt", "square", "cube", "exp", "log")

    def __post_init__(self) -> None:
        for op in self.binary:
            if op not in BINARY:
                raise ValueError(f"unknown binary operator {op!r}")
        for op in self.unary:
            if op not in UNARY:
                raise ValueError(f"unknown unary operator {op!r}")

    @property
    def label(self) -> str:
        return "{" + ", ".join(self.binary + self.unary) + "}"

    def without(self, *ops: str) -> OperatorSet:
        return OperatorSet(
            tuple(o for o in self.binary if o not in ops),
            tuple(o for o in self.unary if o not in ops),
        )

    def plus(self, *ops: str) -> OperatorSet:
        b = self.binary + tuple(o for o in ops if o in BINARY and o not in self.binary)
        u = self.unary + tuple(o for o in ops if o in UNARY and o not in self.unary)
        return OperatorSet(b, u)


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One node. `op` is an operator name, "x" (a feature) or "c" (a constant)."""

    op: str
    children: tuple[Node, ...] = ()
    feature: int = 0
    value: float = 1.0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __str__(self) -> str:
        return to_string(self)


def var(i: int = 0) -> Node:
    return Node("x", feature=i)


def const(v: float = 1.0) -> Node:
    return Node("c", value=float(v))


def un(op: str, a: Node) -> Node:
    return Node(op, (a,))


def bi(op: str, a: Node, b: Node) -> Node:
    return Node(op, (a, b))


def complexity(t: Node) -> int:
    """Node count, PySR's default complexity (every node costs 1)."""
    return 1 + sum(complexity(c) for c in t.children)


def depth(t: Node) -> int:
    return 0 if t.is_leaf else 1 + max(depth(c) for c in t.children)


def n_constants(t: Node) -> int:
    if t.op == "c":
        return 1
    return sum(n_constants(c) for c in t.children)


def has_variable(t: Node) -> bool:
    return t.op == "x" or any(has_variable(c) for c in t.children)


def constants(t: Node) -> list[float]:
    """Leaf constants in pre-order."""
    if t.op == "c":
        return [t.value]
    out: list[float] = []
    for c in t.children:
        out.extend(constants(c))
    return out


def with_constants(t: Node, values) -> Node:
    """A copy of `t` whose constants, in pre-order, are `values`."""
    it = iter(np.asarray(values, float).ravel().tolist())

    def rec(n: Node) -> Node:
        if n.op == "c":
            return replace(n, value=next(it))
        if n.is_leaf:
            return n
        return replace(n, children=tuple(rec(c) for c in n.children))

    return rec(t)


def evaluate(t: Node, X: np.ndarray) -> np.ndarray:
    """Evaluate on rows of X (n, d). Invalid arithmetic returns NaN/inf."""
    X = np.atleast_2d(np.asarray(X, float))
    with np.errstate(all="ignore"):
        out = compiled(t)(X, np.asarray(constants(t), float))
    return np.broadcast_to(np.asarray(out, float), (X.shape[0],))


def _eval(t: Node, X: np.ndarray) -> np.ndarray:
    if t.op == "x":
        return X[:, t.feature]
    if t.op == "c":
        return np.full(X.shape[0], t.value)
    if len(t.children) == 1:
        return UNARY[t.op](_eval(t.children[0], X))
    return BINARY[t.op](_eval(t.children[0], X), _eval(t.children[1], X))


def to_string(t: Node, names: list[str] | None = None, digits: int = 7) -> str:
    """A sympy-parseable string."""
    if t.op == "x":
        return names[t.feature] if names else f"x{t.feature}"
    if t.op == "c":
        # always a float literal, so sympy keeps it a free constant (Float)
        # rather than an Integer that a form check would treat as fixed
        txt = f"{t.value:.{digits}g}"
        if not any(ch in txt for ch in ".eni"):
            txt += ".0"
        return f"({txt})"
    if len(t.children) == 1:
        return _UNARY_FMT[t.op].format(to_string(t.children[0], names, digits))
    a = to_string(t.children[0], names, digits)
    b = to_string(t.children[1], names, digits)
    if t.op == "pow":
        return f"(({a})**({b}))"
    return f"({a} {t.op} {b})"


def structure(t: Node) -> str:
    """The form without constant values: two trees with the same structure
    differ only in what the optimiser is for."""
    if t.op == "x":
        return f"x{t.feature}"
    if t.op == "c":
        return "c"
    return t.op + "(" + ",".join(structure(c) for c in t.children) + ")"


def to_sympy(t: Node, names: list[str] | None = None):
    import sympy

    return sympy.sympify(to_string(t, names))


def simplified(t: Node, names: list[str] | None = None, digits: int = 5) -> str:
    """Readable form: sympy-simplified, constants rounded."""
    import sympy

    try:
        e = sympy.simplify(to_sympy(t, names))
        e = e.xreplace({n: sympy.Float(n, digits) for n in e.atoms(sympy.Float)})
        return str(e)
    except Exception:
        return to_string(t, names, digits)


# ---------------------------------------------------------------------------
# addressing (for the genetic operators)
# ---------------------------------------------------------------------------


def paths(t: Node, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    """Every node's address, pre-order."""
    yield prefix
    for i, c in enumerate(t.children):
        yield from paths(c, (*prefix, i))


def get(t: Node, path: tuple[int, ...]) -> Node:
    for i in path:
        t = t.children[i]
    return t


def put(t: Node, path: tuple[int, ...], sub: Node) -> Node:
    """A copy of `t` with the subtree at `path` replaced by `sub`."""
    if not path:
        return sub
    i = path[0]
    kids = list(t.children)
    kids[i] = put(kids[i], path[1:], sub)
    return replace(t, children=tuple(kids))


# ---------------------------------------------------------------------------
# loss and constant fitting
# ---------------------------------------------------------------------------


def loss(t: Node, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted mean squared error; inf for a non-finite prediction."""
    p = evaluate(t, X)
    if p.shape != y.shape or not np.all(np.isfinite(p)):
        return float("inf")
    r = p - y
    with np.errstate(all="ignore"):
        val = float(np.mean(w * r * r)) if w is not None else float(np.mean(r * r))
    return val if np.isfinite(val) else float("inf")


# Code generation: a tree becomes one Python expression over numpy calls,
# compiled once per structure. Recursive evaluation rebuilds the tree for every
# trial constant vector and made constant fitting ~30 ms per tree -- too slow
# for an exhaustive search over 10^4-10^5 trees.
_GLOBALS = {
    "_sqrt": np.sqrt,
    "_sq": np.square,
    "_exp": np.exp,
    "_log": np.log,
    "_sin": np.sin,
    "_cos": np.cos,
    "_pow": np.power,
    "_cube": lambda a: a * a * a,
}
_SRC_UNARY = {
    "neg": "(-{0})",
    "sqrt": "_sqrt({0})",
    "square": "_sq({0})",
    "cube": "_cube({0})",
    "exp": "_exp({0})",
    "log": "_log({0})",
    "inv": "(1.0/{0})",
    "sin": "_sin({0})",
    "cos": "_cos({0})",
}
_COMPILED: dict[str, Callable] = {}


def _source(t: Node, k: list[int]) -> str:
    if t.op == "x":
        return f"X[:, {t.feature}]"
    if t.op == "c":
        k[0] += 1
        return f"c[{k[0] - 1}]"
    if len(t.children) == 1:
        return _SRC_UNARY[t.op].format(_source(t.children[0], k))
    a = _source(t.children[0], k)
    b = _source(t.children[1], k)
    if t.op == "pow":
        return f"_pow({a}, {b})"
    return f"({a} {t.op} {b})"


def compiled(t: Node) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """f(X, c) for the structure of `t`, constants passed in pre-order."""
    key = structure(t)
    fn = _COMPILED.get(key)
    if fn is None:
        if len(_COMPILED) > 200_000:
            _COMPILED.clear()
        fn = eval("lambda X, c: " + _source(t, [0]), dict(_GLOBALS))
        _COMPILED[key] = fn
    return fn


def fit_constants(
    t: Node,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    *,
    method: str = "lm",
    restarts: int = 1,
    rng: np.random.Generator | None = None,
    max_nfev: int = 100,
    good_enough: float = 0.0,
) -> tuple[Node, float]:
    """Best constants for a fixed form, and the loss they reach.

    method "lm"   Levenberg-Marquardt on the weighted residuals (MINPACK
                  lmdif via `scipy.optimize.leastsq`, finite-difference
                  Jacobian) -- the natural choice for a least-squares loss;
    method "bfgs" quasi-Newton on the scalar loss, which is what
                  SymbolicRegression.jl uses by default (Newton when there is
                  a single constant).

    The loss surface in the constants is non-convex whenever a constant sits
    inside a nonlinearity (exp(c*x)), so a single local search can stall.
    Restarts perturb the start multiplicatively, the scheme
    SymbolicRegression.jl uses (`x0 * (1 + eps/2)`, eps ~ N(0,1)). A restart
    is skipped once the loss is below `good_enough`.
    """
    k = n_constants(t)
    X = np.atleast_2d(np.asarray(X, float))
    base = loss(t, X, y, w)
    if k == 0:
        return t, base
    rng = rng or np.random.default_rng(0)
    sw = np.sqrt(w) if w is not None else np.ones_like(y)
    fn = compiled(t)

    def resid(c):
        r = sw * (fn(X, c) - y)
        if np.shape(r) != y.shape:
            r = np.broadcast_to(r, y.shape)
        return np.where(np.isfinite(r), r, 1e100)

    def mse(c):
        r = resid(c)
        return float(np.mean(r * r))

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_c = _search_constants(
            t, resid, mse, base, k, method, restarts, rng, max_nfev, good_enough, len(y)
        )
    fitted = with_constants(t, best_c)
    return fitted, loss(fitted, X, y, w)


def _search_constants(
    t, resid, mse, base, k, method, restarts, rng, max_nfev, good_enough, n
):
    from scipy.optimize import leastsq, minimize

    best_c = np.asarray(constants(t), float)
    best = base if np.isfinite(base) else float("inf")
    c_start = best_c.copy()
    for attempt in range(restarts + 1):
        if attempt == 0:
            c0 = c_start
        else:
            if best <= good_enough:
                break
            eps = rng.standard_normal(k)
            c0 = np.where(c_start == 0, eps, c_start * (1.0 + 0.5 * eps))
        try:
            if method == "lm" and n >= k:
                c = leastsq(resid, c0, maxfev=max_nfev)[0]
            else:
                c = minimize(mse, c0, method="BFGS", options={"maxiter": max_nfev}).x
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, TypeError):
            continue
        c = np.atleast_1d(np.asarray(c, float))
        val = mse(c)
        if np.all(np.isfinite(c)) and val < best:
            best, best_c = val, c
    return best_c


# ---------------------------------------------------------------------------
# the Pareto front and model selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    tree: Node
    loss: float
    complexity: int

    @classmethod
    def of(cls, tree: Node, loss_value: float) -> Candidate:
        return cls(tree, float(loss_value), complexity(tree))

    def expression(self, names: list[str] | None = None) -> str:
        return to_string(self.tree, names)


def pareto_front(cands) -> list[Candidate]:
    """Best candidate at each complexity, keeping only those that beat every
    simpler one. Sorted by complexity; loss strictly decreasing."""
    best: dict[int, Candidate] = {}
    for c in cands:
        if not np.isfinite(c.loss):
            continue
        if c.complexity not in best or c.loss < best[c.complexity].loss:
            best[c.complexity] = c
    front: list[Candidate] = []
    for k in sorted(best):
        # a relative margin, so a longer spelling of the same function
        # (log(sqrt(c)) for c) does not enter on a rounding difference
        if not front or best[k].loss < front[-1].loss * (1.0 - 1e-9):
            front.append(best[k])
    return front


def scores(front: list[Candidate]) -> list[float]:
    """PySR's score: -d log(loss) / d complexity between neighbours on the
    front (`calculate_scores` in pysr/sr.py). The first entry is 0. A loss of
    exactly zero gets +inf, as there."""
    out = [0.0]
    for prev, cur in itertools.pairwise(front):
        if cur.loss > 0:
            out.append(
                -float(np.log(cur.loss / prev.loss))
                / (cur.complexity - prev.complexity)
            )
        else:
            out.append(float("inf"))
    return out


def select(front: list[Candidate], rule: str = "best") -> Candidate:
    """Pick one expression off the front, with PySR's three rules
    (`idx_model_selection` in pysr/sr.py):

    accuracy   lowest loss;
    score      highest score;
    best       highest score among those with loss <= 1.5 x the lowest.
    """
    if not front:
        raise ValueError("empty front")
    s = scores(front)
    if rule == "accuracy":
        return min(front, key=lambda c: c.loss)
    if rule == "score":
        return front[int(np.argmax(s))]
    if rule == "best":
        lmin = min(c.loss for c in front)
        ok = [i for i, c in enumerate(front) if c.loss <= 1.5 * lmin]
        return front[max(ok, key=lambda i: s[i])]
    raise ValueError(f"unknown selection rule {rule!r}")


def noise_floor(sigma_rel: float, n: int, k: int = 3) -> float:
    """The mean squared relative error a correct law reaches on data with
    relative noise `sigma_rel`, plus k standard deviations of chi^2/n.

    Used as a stopping rule by every search here: the searches know the
    measurement error (as a physicist does), not the law. A floor of 1e-6
    relative stands in for "noise-free" so rounding in the constants is not
    mistaken for a wrong form.
    """
    s2 = max(sigma_rel, 1e-6) ** 2
    return s2 * (1.0 + k * np.sqrt(2.0 / max(n, 1)))
