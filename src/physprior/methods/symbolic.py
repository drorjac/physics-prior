"""Symbolic regression: the only arm that can return a law it was not given.

PySR (Cranmer 2023) searches closed forms over a chosen operator set. Runs are
cached on disk keyed by the data and the search configuration, because a
search costs tens of seconds and the sweeps repeat many of them.

Two things are recorded that an RMSE cannot show:

`expression`    the closed form itself, simplified by sympy;
`recovered`     whether it matches the target law, judged by a track-supplied
                predicate -- e.g. "is there a power of f with exponent
                11/3 +- 0.1?". Getting the right curve and getting the right
                law are different achievements and are scored separately.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from physprior.config import get_settings

from .base import Fit


def _cache_dir() -> Path:
    """Where finished searches are kept. Resolved per call so a test can point
    it at a temporary directory."""
    path = get_settings().sr_cache_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_BINARY = ["+", "-", "*", "/"]
DEFAULT_UNARY = ["square", "cube", "sqrt", "exp", "log"]


def _key(X: np.ndarray, y: np.ndarray, cfg: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, float)).tobytes())
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()[:24]


def fit_sr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    niterations: int = 60,
    maxsize: int = 20,
    seed: int = 0,
    complexity_of_constants: int = 1,
    model_selection: str = "best",
    constraints: dict | None = None,
    use_cache: bool = True,
    name: str = "sr",
) -> Fit:
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    if X.shape[0] != len(y):
        X = X.T
    binary_operators = binary_operators or DEFAULT_BINARY
    unary_operators = unary_operators if unary_operators is not None else DEFAULT_UNARY
    if "^" in binary_operators and constraints is None:
        # arbitrary base, but the exponent must stay a constant or a variable
        constraints = {"^": (-1, 1)}
    cfg = {
        "bin": binary_operators,
        "un": unary_operators,
        "it": niterations,
        "ms": maxsize,
        "seed": seed,
        "coc": complexity_of_constants,
        "sel": model_selection,
        "con": constraints or {},
        "fn": feature_names or [],
    }
    path = _cache_dir() / f"{_key(X, y, cfg)}.pkl"
    if use_cache and path.exists():
        d = pickle.loads(path.read_bytes())
        return _to_fit(d, name)

    from pysr import PySRRegressor

    t0 = time.time()
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        maxsize=maxsize,
        complexity_of_constants=complexity_of_constants,
        model_selection=model_selection,
        constraints=constraints,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model.fit(X, y, variable_names=feature_names)
    eqs = model.equations_
    best = model.sympy()
    d = {
        "expression": str(best),
        "latex": model.latex(),
        "table": eqs[["complexity", "loss", "equation"]].to_dict("records"),
        "lambda_src": str(best),
        "feature_names": feature_names or [f"x{i}" for i in range(X.shape[1])],
        "seconds": time.time() - t0,
        "n_free": int(
            eqs["complexity"].iloc[
                model.equations_.index.get_loc(model.equations_.index[-1])
            ]
        )
        if len(eqs)
        else 0,
    }
    if use_cache:
        path.write_bytes(pickle.dumps(d))
    return _to_fit(d, name)


def _to_fit(d: dict, name: str) -> Fit:
    import sympy

    names = d["feature_names"]
    syms = sympy.symbols(names)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    expr = sympy.sympify(d["expression"])
    fn = sympy.lambdify(syms, expr, "numpy")

    def predict(Xq: np.ndarray) -> np.ndarray:
        Xq = np.atleast_2d(np.asarray(Xq, float))
        if Xq.shape[1] != len(names):
            Xq = Xq.T
        out = fn(*[Xq[:, i] for i in range(len(names))])
        return np.broadcast_to(np.asarray(out, float), (Xq.shape[0],)).copy()

    return Fit(
        name=name,
        predict=predict,
        expression=d["expression"],
        n_free=int(d.get("n_free", 0)),
        seconds=float(d.get("seconds", 0.0)),
        extra={"latex": d.get("latex"), "pareto": d.get("table")},
    )


def power_law_exponent(
    fit_or_expr, var: str, x_range: tuple[float, float], tol: float = 0.02
) -> float | None:
    """The exponent of `var`, measured rather than parsed.

    A discovered power law rarely comes back in the form `c*f**p`; PySR is as
    likely to write `f**2.01 * f**1.65 * f / f`. So the exponent is measured
    the way a physicist would: the slope of log y against log x, checked for
    constancy across the range. `None` means "not a power law here".
    """
    import numpy as _np
    import sympy

    expr = (
        fit_or_expr.expression
        if hasattr(fit_or_expr, "expression")
        else str(fit_or_expr)
    )
    try:
        x = sympy.Symbol(var)
        fn = sympy.lambdify(x, sympy.sympify(expr), "numpy")
        xs = _np.geomspace(max(x_range[0], 1e-12), x_range[1], 64)
        ys = _np.asarray(fn(xs), float)
        if not _np.all(_np.isfinite(ys)) or _np.any(ys == 0):
            return None
        # Bound-state energies are negative. A power law in |y| is still a
        # power law; what disqualifies an expression is a SIGN CHANGE, not a
        # sign. An earlier version required y > 0 and silently reported "not a
        # power law" for every Coulomb spectrum.
        if _np.any(ys > 0) and _np.any(ys < 0):
            return None
        ys = _np.abs(ys)
        slopes = _np.diff(_np.log(ys)) / _np.diff(_np.log(xs))
        if _np.ptp(slopes) > tol * max(1.0, abs(float(_np.mean(slopes)))):
            return None  # curved in log-log: not a pure power law
        return float(_np.mean(slopes))
    except Exception:
        return None
