"""The measured experiments behind docs/theory/symbolic_regression.md.

    run(quick=False, parts=None)

Parts (each writes its own results/symbolic/<name>.csv|json, so they can be
run in separate processes):

    growth       size of the search space vs tree size, depth and vocabulary;
                 time for exhaustive search to reach a law
    tune         GP parsimony chosen on the TUNING seeds 3/7/19
    noise        recovery vs noise: GP, exhaustive, PySR, four laws
    budget       recovery vs number of samples: GP, exhaustive
    vocabulary   Planck's law with and without exp in the vocabulary
    pareto       the loss-complexity front and the score, on one dataset
    sindy        SINDy on a damped oscillator and a pendulum vs noise, with
                 the derivative convergence and noise-amplification studies
    packages     checks of what curve_fit, autograd and Adam actually do

Law recovery is measured, not parsed, as elsewhere in the repo: the selected
expression's FORM (constants freed) is refitted to the noise-free law on a
grid reaching well outside the data. A form that can represent the law
there to 1e-4 relative RMS has recovered it; a curve that only fits the
data has not.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.constants import AU_M, GM_SUN, RYDBERG_H_CM
from physprior.io import load_json, load_table, save_json, save_table
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS
from physprior.units import require

from .enumerate import (
    count_by_depth,
    count_by_size,
    count_enumerated,
    exhaustive_search,
)
from .expressions import (
    Candidate,
    OperatorSet,
    noise_floor,
    scores,
    select,
    to_sympy,
)
from .gp import GPConfig, gp_search

AREA = "symbolic"
DEFAULT_OPS = (
    OperatorSet()
)  # the repo's PySR defaults: + - * / sqrt square cube exp log
FORM_TOL = 1e-4  # relative RMS of the refitted form against the exact law
EXTRAP_TOL = 0.05  # worst relative error of the as-fitted expression off the data

# ---------------------------------------------------------------------------
# the laws
# ---------------------------------------------------------------------------

# Rydberg constant for hydrogen in units of 1e5 cm^-1 (constants.RYDBERG_H_CM,
# CODATA22), so the target is O(1).
R_H_1E5 = RYDBERG_H_CM / 1e5
# GM_sun / AU^2 in mm/s^2 (IAU2015 values in constants.py): the Sun's pull at
# 1 AU, ~5.93 mm/s^2.
K_SUN_MM = GM_SUN / AU_M**2 * 1e3


@dataclass(frozen=True)
class Law:
    name: str
    formula: str
    truth: Callable[[np.ndarray], np.ndarray]
    sample: Callable[[int, np.random.Generator], np.ndarray]
    check: np.ndarray  # extended grid for the form check
    band: tuple[float, float]  # where the data are


def _kepler_sample(n, rng):
    return np.sort(np.exp(rng.uniform(np.log(0.39), np.log(5.2), n)))


LAWS: dict[str, Law] = {
    "kepler": Law(
        "kepler",
        "P = a^{3/2}  (years, AU)",
        lambda a: a**1.5,
        _kepler_sample,
        np.geomspace(0.2, 40.0, 200),
        (0.39, 5.2),
    ),
    "bohr": Law(
        "bohr",
        "nu = R_H (1 - 1/n^2)  (1e5 cm^-1)",
        lambda n: R_H_1E5 * (1.0 - 1.0 / n**2),
        lambda k, rng: np.arange(2.0, 2.0 + k),
        np.geomspace(2.0, 200.0, 200),
        (2.0, 2.0),  # upper end depends on n; see make_data
    ),
    "inverse_square": Law(
        "inverse_square",
        "g = GM_sun / r^2  (mm/s^2, AU)",
        lambda r: K_SUN_MM / r**2,
        lambda k, rng: np.sort(rng.uniform(0.39, 1.52, k)),
        np.geomspace(0.1, 40.0, 200),
        (0.39, 1.52),
    ),
    "planck": Law(
        "planck",
        "B = x^3 / (e^x - 1)  (x = h nu / k T)",
        lambda x: x**3 / np.expm1(x),
        lambda k, rng: np.sort(rng.uniform(0.5, 8.0, k)),
        np.geomspace(0.05, 15.0, 200),
        (0.5, 8.0),
    ),
}


def make_data(law: Law, n: int, noise: float, seed: int):
    """x, noisy y, weights 1/y^2 (relative errors), clean y."""
    rng = np.random.default_rng(seed)
    x = law.sample(n, rng)
    y0 = law.truth(x)
    y = y0 * (1.0 + noise * rng.standard_normal(len(x)))
    require(bool(np.all(np.isfinite(y)) and np.all(y != 0)), f"{law.name}: bad targets")
    return x, y, 1.0 / y**2, y0


# ---------------------------------------------------------------------------
# judging an expression
# ---------------------------------------------------------------------------


def form_check(expr, law: Law, var: str = "x0") -> dict:
    """Free every float in `expr`, refit to the exact law on `law.check`.

    Two questions, both needed for "recovered":

    form_ok     can this form represent the law, with some constants, to
                FORM_TOL relative RMS over the whole grid? (A bloated form
                that nests the law passes this.)
    extrap_dev  how wrong are the expression's predictions, with the
                constants it was actually fitted with, anywhere on the grid,
                including far outside the data? Must be <= EXTRAP_TOL.
    """
    import sympy
    from scipy.optimize import least_squares

    out: dict[str, Any] = {
        "form_err": float("inf"),
        "form_ok": False,
        "extrap_dev": float("inf"),
        "recovered": False,
    }
    try:
        e = sympy.sympify(expr) if isinstance(expr, str) else expr
        x = sympy.Symbol(var)
        floats = sorted(e.atoms(sympy.Float), key=str)
        ps = sympy.symbols(f"p0:{len(floats)}") if floats else ()
        ef = e.xreplace(dict(zip(floats, ps, strict=True)))
        f = sympy.lambdify((x, *ps), ef, "numpy")
        xs = law.check
        yt = law.truth(xs)
        p0 = np.array([float(v) for v in floats])

        def rel(p):
            with np.errstate(all="ignore"):
                v = np.broadcast_to(np.asarray(f(xs, *p), float), xs.shape)
            r = (v - yt) / yt
            return np.where(np.isfinite(r), r, 1e6)

        r0 = rel(p0)
        out["extrap_dev"] = float(np.max(np.abs(r0)))
        best = float(np.sqrt(np.mean(r0**2)))
        if len(p0):
            with np.errstate(all="ignore"):
                res = least_squares(rel, p0, method="lm", max_nfev=2000)
            best = min(best, float(np.sqrt(np.mean(res.fun**2))))
        out["form_err"] = best
        out["form_ok"] = bool(best < FORM_TOL)
        out["recovered"] = bool(out["form_ok"] and out["extrap_dev"] <= EXTRAP_TOL)
    except Exception as exc:  # an unparseable expression has not recovered anything
        out["error"] = type(exc).__name__
    return out


def _cand_rows(front: list[Candidate]) -> list[dict]:
    s = scores(front)
    return [
        {
            "complexity": c.complexity,
            "loss": c.loss,
            "score": sc,
            "expression": c.expression(),
        }
        for c, sc in zip(front, s, strict=True)
    ]


def _judge(front: list[Candidate], law: Law) -> dict:
    if not front:
        return {"expression": None, "recovered": False, "on_front": False}
    sel = select(front, "best")
    chk = form_check(to_sympy(sel.tree), law)
    on_front = chk["recovered"] or any(
        form_check(to_sympy(c.tree), law)["recovered"] for c in front if c is not sel
    )
    return {
        "expression": sel.expression(),
        "complexity": sel.complexity,
        "loss": sel.loss,
        "form_err": chk["form_err"],
        "form_ok": chk["form_ok"],
        "extrap_dev": chk["extrap_dev"],
        "recovered": chk["recovered"],
        "on_front": bool(on_front),
    }


# ---------------------------------------------------------------------------
# the three searchers, one calling convention
# ---------------------------------------------------------------------------


def run_gp(x, y, w, ops, seed, stop, config: GPConfig | None = None):
    r = gp_search(x[:, None], y, w, ops=ops, config=config, seed=seed, stop_loss=stop)
    return (
        r.front,
        {
            "seconds": r.seconds,
            "n_evaluated": r.n_evaluated,
            "n_structures": r.n_structures,
            "reached_floor": r.reached_floor_at is not None,
            "generations": len(r.history),
        },
        r,
    )


def run_exhaustive(x, y, w, ops, seed, stop, max_size=7, time_budget=None):
    r = exhaustive_search(
        x[:, None],
        y,
        w,
        ops=ops,
        max_size=max_size,
        stop_loss=stop,
        seed=seed,
        time_budget=time_budget,
    )
    return (
        r.front,
        {
            "seconds": r.seconds,
            "n_evaluated": r.n_evaluated,
            "size_reached": r.size_reached,
            "reached_floor": r.reached_floor,
            "timed_out": r.timed_out,
        },
        r,
    )


PYSR_CFG = {"niterations": 40, "populations": 15, "maxsize": 15}


def pysr_available() -> bool:
    if os.environ.get("PHYSPRIOR_NO_JULIA") == "1":
        return False
    try:
        import pysr  # noqa: F401
    except Exception:
        return False
    return True


def run_pysr(x, y, w, ops: OperatorSet, seed: int, cfg: dict | None = None):
    """A short PySR search, cached on disk by data + configuration.

    Returns the front re-scored with THIS module's loss (mean w r^2) on the
    same data, so the three searchers' fronts share an axis, plus PySR's own
    choice under model_selection="best".
    """
    import sympy

    cfg = {**PYSR_CFG, **(cfg or {})}
    full = {**cfg, "bin": list(ops.binary), "un": list(ops.unary), "seed": seed}
    h = hashlib.sha256()
    for a in (x, y, w):
        h.update(np.ascontiguousarray(a, float).tobytes())
    h.update(json.dumps(full, sort_keys=True).encode())
    cdir = get_settings().cache_dir / "sr_theory"
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{h.hexdigest()[:24]}.json"
    if path.exists():
        d = json.loads(path.read_text())
    else:
        from pysr import PySRRegressor

        t0 = time.time()
        model = PySRRegressor(
            niterations=cfg["niterations"],
            populations=cfg["populations"],
            maxsize=cfg["maxsize"],
            binary_operators=[("^" if o == "pow" else o) for o in ops.binary],
            unary_operators=list(ops.unary),
            model_selection="best",
            deterministic=True,
            parallelism="serial",
            random_state=seed,
            verbosity=0,
            progress=False,
            temp_equation_file=True,
        )
        model.fit(x[:, None], y, weights=w, variable_names=["x0"])
        eqs = model.equations_
        d = {
            "seconds": time.time() - t0,
            "selected": str(model.sympy()),
            "front": [
                {
                    "complexity": int(r["complexity"]),
                    "pysr_loss": float(r["loss"]),
                    "expression": str(r["sympy_format"]),
                }
                for _, r in eqs.iterrows()
            ],
            "config": full,
        }
        path.write_text(json.dumps(d, indent=1))
    xs = sympy.Symbol("x0")
    for row in d["front"]:
        try:
            f = sympy.lambdify(xs, sympy.sympify(row["expression"]), "numpy")
            with np.errstate(all="ignore"):
                p = np.broadcast_to(np.asarray(f(x), float), x.shape)
            row["loss"] = float(np.mean(w * (p - y) ** 2))
        except Exception:
            row["loss"] = float("inf")
    return d


def _judge_pysr(d: dict, law: Law) -> dict:
    chk = form_check(d["selected"], law)
    sel = next((r for r in d["front"] if r["expression"] == d["selected"]), None)
    on_front = chk["recovered"] or any(
        form_check(r["expression"], law)["recovered"] for r in d["front"]
    )
    return {
        "expression": d["selected"],
        "complexity": sel["complexity"] if sel else None,
        "loss": sel["loss"] if sel else None,
        "form_err": chk["form_err"],
        "form_ok": chk["form_ok"],
        "extrap_dev": chk["extrap_dev"],
        "recovered": chk["recovered"],
        "on_front": bool(on_front),
        "seconds": d["seconds"],
    }


def _gp_config() -> GPConfig:
    """The GP configuration, with parsimony as chosen on the tuning seeds."""
    try:
        chosen = load_json(AREA, "tune_gp")["chosen_parsimony"]
    except (FileNotFoundError, KeyError):
        chosen = GPConfig().parsimony
    return GPConfig(parsimony=float(chosen))


# ---------------------------------------------------------------------------
# (a) the size of the search space
# ---------------------------------------------------------------------------

VOCAB_LADDER = [
    OperatorSet(("+", "*"), ()),
    OperatorSet(("+", "-", "*", "/"), ()),
    OperatorSet(("+", "-", "*", "/"), ("square",)),
    OperatorSet(("+", "-", "*", "/"), ("square", "sqrt")),
    OperatorSet(("+", "-", "*", "/"), ("square", "sqrt", "cube")),
    OperatorSet(("+", "-", "*", "/"), ("square", "sqrt", "cube", "exp")),
    OperatorSet(("+", "-", "*", "/"), ("square", "sqrt", "cube", "exp", "log")),
    OperatorSet(("+", "-", "*", "/"), ("square", "sqrt", "cube", "exp", "log", "sin")),
    OperatorSet(
        ("+", "-", "*", "/"), ("square", "sqrt", "cube", "exp", "log", "sin", "cos")
    ),
]


def study_growth(quick: bool = False) -> dict:
    max_size = 12 if not quick else 8
    rows = []
    for ops in VOCAB_LADDER:
        raw = count_by_size(max_size, 2, len(ops.unary), len(ops.binary))
        for s, n in enumerate(raw, start=1):
            rows.append(
                {
                    "n_binary": len(ops.binary),
                    "n_unary": len(ops.unary),
                    "n_ops": len(ops.binary) + len(ops.unary),
                    "vocabulary": ops.label,
                    "size": s,
                    "n_trees_raw": float(n),
                }
            )
    counts = pd.DataFrame(rows)
    # pruned counts by enumeration, for the default vocabulary and a small one
    pr_rows = []
    pr_max = 7 if not quick else 5
    for ops in (VOCAB_LADDER[1], DEFAULT_OPS):
        pr = count_enumerated(ops, pr_max)
        raw = count_by_size(pr_max, 2, len(ops.unary), len(ops.binary))
        for s, (a, b) in enumerate(zip(pr, raw, strict=True), start=1):
            pr_rows.append(
                {
                    "vocabulary": ops.label,
                    "size": s,
                    "n_trees_raw": b,
                    "n_trees_pruned": a,
                }
            )
    depth = [
        {"depth": d, "n_trees_le_depth": float(n)}
        for d, n in enumerate(count_by_depth(5, 2, len(DEFAULT_OPS.unary), 4))
    ]
    save_table(counts, AREA, "growth_counts")
    save_table(pd.DataFrame(pr_rows), AREA, "growth_pruned")
    save_table(pd.DataFrame(depth), AREA, "growth_depth")

    # time for exhaustive search to reach each law, noise-free
    ttf = []
    names = ["kepler", "inverse_square", "bohr"] + ([] if quick else ["planck"])
    for name in names:
        law = LAWS[name]
        for seed in REPORT_SEEDS[: (1 if quick or name == "planck" else 3)]:
            x, y, w, _ = make_data(law, 24, 0.0, seed)
            front, info, _ = run_exhaustive(
                x, y, w, DEFAULT_OPS, seed, noise_floor(0.0, len(x)), max_size=7
            )
            ttf.append(
                {
                    "law": name,
                    "seed": seed,
                    "vocabulary": DEFAULT_OPS.label,
                    **info,
                    **_judge(front, law),
                }
            )
            print(
                f"  [growth] {name} seed {seed}: {info['seconds']:.1f}s "
                f"{info['n_evaluated']} trees",
                flush=True,
            )
    save_table(pd.DataFrame(ttf), AREA, "time_to_find")

    # the same law (Bohr, size 6) against a growing vocabulary
    vrows = []
    ladder = VOCAB_LADDER[2:] if not quick else VOCAB_LADDER[2:4]
    law = LAWS["bohr"]
    x, y, w, _ = make_data(law, 24, 0.0, REPORT_SEEDS[0])
    for ops in ladder:
        front, info, _ = run_exhaustive(
            x, y, w, ops, REPORT_SEEDS[0], noise_floor(0.0, len(x)), max_size=6
        )
        vrows.append(
            {
                "law": "bohr",
                "n_ops": len(ops.binary) + len(ops.unary),
                "vocabulary": ops.label,
                **info,
                **_judge(front, law),
            }
        )
        print(
            f"  [growth] bohr, {vrows[-1]['n_ops']} ops: {info['seconds']:.1f}s",
            flush=True,
        )
    save_table(pd.DataFrame(vrows), AREA, "time_vs_vocabulary")
    return {"n_counts": len(counts)}


# ---------------------------------------------------------------------------
# tuning the GP (tuning seeds only)
# ---------------------------------------------------------------------------

# 0.1 and 1.0 were added after the best value came out at the edge of the grid
# (a choice at the boundary of its range has not converged)
PARSIMONY_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


def study_tune(quick: bool = False) -> dict:
    """Parsimony chosen by recovery rate (then time) at 1 % noise, n = 24, on
    seeds 3/7/19. Nothing here looks at a reporting seed."""
    rows = []
    grid = PARSIMONY_GRID if not quick else PARSIMONY_GRID[:2]
    seeds = TUNE_SEEDS if not quick else TUNE_SEEDS[:1]
    # runs are deterministic given (parsimony, law, seed): reuse finished ones.
    # A parsimony value with only some of its runs is re-run whole, so its
    # partial rows are dropped first rather than counted twice.
    done = pd.DataFrame()
    if not quick:
        try:
            done = load_table(AREA, "tune_gp_runs")
            done = done[done.parsimony.isin(grid) & done.seed.isin(seeds)]
            n_runs = done.groupby("parsimony")["seed"].transform("size")
            done = done[n_runs == len(LAWS) * len(seeds)]
            rows = done.to_dict("records")
        except FileNotFoundError:
            pass
    for p in grid:
        if len(done) and (done.parsimony == p).any():
            continue
        cfg = GPConfig(parsimony=p)
        for name, law in LAWS.items():
            for seed in seeds:
                x, y, w, _ = make_data(law, 24, 0.01, seed)
                front, info, _ = run_gp(
                    x, y, w, DEFAULT_OPS, seed, noise_floor(0.01, 24), cfg
                )
                rows.append(
                    {
                        "parsimony": p,
                        "law": name,
                        "seed": seed,
                        **info,
                        **_judge(front, law),
                    }
                )
        print(f"  [tune] parsimony {p}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("parsimony").agg(
        rate=("recovered", "mean"), sec=("seconds", "mean")
    )
    chosen = float(agg.sort_values(["rate", "sec"], ascending=[False, True]).index[0])
    # A quick run's choice comes from a cut-down grid and one seed. It is not
    # saved, so `_gp_config` keeps reading the full tuning result.
    if quick:
        return {"chosen_parsimony": chosen}
    save_table(df, AREA, "tune_gp_runs")
    save_json(
        {
            "chosen_parsimony": chosen,
            "grid": list(grid),
            "seeds": list(seeds),
            "rate": agg["rate"].to_dict(),
            "seconds": agg["sec"].to_dict(),
        },
        AREA,
        "tune_gp",
    )
    return {"chosen_parsimony": chosen}


# ---------------------------------------------------------------------------
# (b) recovery vs noise and vs sample size
# ---------------------------------------------------------------------------

NOISE_LEVELS = (0.0, 1e-3, 1e-2, 1e-1)
# n = 24 at 1 % noise is already in the noise sweep; it is not run twice
BUDGETS = (6, 12, 48)
PYSR_NOISE = (0.0, 1e-2, 1e-1)
N_DEFAULT = 24


def _sweep(kind: str, quick: bool, methods=("gp", "exhaustive")) -> pd.DataFrame:
    cfg = _gp_config()
    levels: tuple[float, ...] = NOISE_LEVELS if kind == "noise" else BUDGETS
    if quick:
        levels = levels[:2]
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    laws = list(LAWS) if not quick else ["kepler", "inverse_square"]
    rows = []
    for name in laws:
        law = LAWS[name]
        for lev in levels:
            noise, n = (lev, N_DEFAULT) if kind == "noise" else (0.01, int(lev))
            for seed in seeds:
                x, y, w, _ = make_data(law, n, noise, seed)
                stop = noise_floor(noise, n)
                base = {"law": name, "noise": noise, "n": n, "seed": seed}
                if "gp" in methods:
                    front, info, _ = run_gp(x, y, w, DEFAULT_OPS, seed, stop, cfg)
                    rows.append({**base, "method": "gp", **info, **_judge(front, law)})
                # exhaustive search to size 7 costs minutes per Planck
                # dataset (size 7, two constants); it is run for Planck only in
                # the vocabulary study
                if "exhaustive" in methods and name != "planck":
                    front, info, _ = run_exhaustive(x, y, w, DEFAULT_OPS, seed, stop)
                    rows.append(
                        {**base, "method": "exhaustive", **info, **_judge(front, law)}
                    )
                if "pysr" in methods and pysr_available():
                    d = run_pysr(x, y, w, DEFAULT_OPS, seed)
                    rows.append({**base, "method": "pysr", **_judge_pysr(d, law)})
            print(f"  [{kind}] {name} level {lev} done", flush=True)
    return pd.DataFrame(rows)


def study_noise(quick: bool = False) -> dict:
    df = _sweep("noise", quick)
    save_table(df, AREA, "recovery_noise")
    return {"rows": len(df)}


def study_budget(quick: bool = False) -> dict:
    df = _sweep("budget", quick)
    save_table(df, AREA, "recovery_budget")
    return {"rows": len(df)}


def study_pysr(quick: bool = False) -> dict:
    """PySR on the noise sweep (a subset of levels: each run costs tens of
    seconds). Skipped, with a note in the results, when Julia is off."""
    if not pysr_available():
        save_json({"skipped": "PySR/Julia not available"}, AREA, "recovery_pysr_meta")
        return {"skipped": True}
    rows = []
    levels = PYSR_NOISE if not quick else PYSR_NOISE[:1]
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    laws = list(LAWS) if not quick else ["kepler"]
    for name in laws:
        law = LAWS[name]
        for noise in levels:
            for seed in seeds:
                x, y, w, _ = make_data(law, N_DEFAULT, noise, seed)
                d = run_pysr(x, y, w, DEFAULT_OPS, seed)
                rows.append(
                    {
                        "law": name,
                        "noise": noise,
                        "n": N_DEFAULT,
                        "seed": seed,
                        "method": "pysr",
                        **_judge_pysr(d, law),
                    }
                )
                print(
                    f"  [pysr] {name} noise {noise} seed {seed}: "
                    f"{rows[-1]['expression']}",
                    flush=True,
                )
    save_table(pd.DataFrame(rows), AREA, "recovery_pysr")
    save_json(
        {"config": PYSR_CFG, "ops": DEFAULT_OPS.label}, AREA, "recovery_pysr_meta"
    )
    return {"rows": len(rows)}


# ---------------------------------------------------------------------------
# the vocabulary lesson (H6)
# ---------------------------------------------------------------------------

VOCAB_NOISE = 1e-3


def study_vocabulary(quick: bool = False) -> dict:
    """Planck's law needs exp. Without it no search can write the law down,
    however long it runs; with it, the question becomes whether the search
    finds it. Same data, same method, only the vocabulary changes."""
    law = LAWS["planck"]
    with_exp = DEFAULT_OPS
    without = DEFAULT_OPS.without("exp", "log")
    cfg = _gp_config()
    rows, curves = [], {}
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    for label, ops in (("without exp", without), ("with exp", with_exp)):
        for seed in seeds:
            x, y, w, _ = make_data(law, N_DEFAULT, VOCAB_NOISE, seed)
            stop = noise_floor(VOCAB_NOISE, N_DEFAULT)
            base = {"vocabulary": label, "ops": ops.label, "seed": seed}
            front, info, _ = run_gp(x, y, w, ops, seed, stop, cfg)
            j = _judge(front, law)
            rows.append({**base, "method": "gp", **info, **j})
            curves[f"gp|{label}|{seed}"] = j["expression"]
            if seed == REPORT_SEEDS[0] and not quick:
                # one exhaustive run per vocabulary: minutes each (see docs)
                front, info, _ = run_exhaustive(x, y, w, ops, seed, stop, max_size=7)
                j = _judge(front, law)
                rows.append({**base, "method": "exhaustive", **info, **j})
                curves[f"exhaustive|{label}|{seed}"] = j["expression"]
            if pysr_available() and not quick:
                d = run_pysr(x, y, w, ops, seed)
                j = _judge_pysr(d, law)
                rows.append({**base, "method": "pysr", **j})
                curves[f"pysr|{label}|{seed}"] = j["expression"]
            print(f"  [vocabulary] {label} seed {seed}", flush=True)
    save_table(pd.DataFrame(rows), AREA, "vocabulary")
    save_json(
        {
            "expressions": curves,
            "noise": VOCAB_NOISE,
            "band": law.band,
            "law": law.formula,
        },
        AREA,
        "vocabulary_curves",
    )
    return {"rows": len(rows)}


# ---------------------------------------------------------------------------
# (c) the Pareto front
# ---------------------------------------------------------------------------

PARETO_LAW, PARETO_NOISE, PARETO_N = "bohr", 1e-2, 16


def study_pareto(quick: bool = False) -> dict:
    """Every searcher's front on one dataset (Bohr, 1 % noise, 16 levels,
    seed 11), with PySR's score along it, and the GP front generation by
    generation."""
    law = LAWS[PARETO_LAW]
    seed = REPORT_SEEDS[0]
    x, y, w, _ = make_data(law, PARETO_N, PARETO_NOISE, seed)
    noise_floor(PARETO_NOISE, PARETO_N)
    rows = []
    cfg = _gp_config()
    # no early stop here: the point is to watch the whole front form
    front, _, r = run_gp(x, y, w, DEFAULT_OPS, seed, None, cfg)
    sel = select(front)
    rows += [
        {"method": "gp", **row, "selected": row["expression"] == sel.expression()}
        for row in _cand_rows(front)
    ]
    evo = [
        {"generation": g, "complexity": c, "loss": lo, "expression": e}
        for g, fr in enumerate(r.fronts)
        for (c, lo, e) in fr
    ]
    save_table(pd.DataFrame(evo), AREA, "pareto_gp_generations")
    if not quick:
        front, _, _ = run_exhaustive(x, y, w, DEFAULT_OPS, seed, None, max_size=7)
        sel = select(front)
        rows += [
            {
                "method": "exhaustive",
                **row,
                "selected": row["expression"] == sel.expression(),
            }
            for row in _cand_rows(front)
        ]
    if pysr_available() and not quick:
        d = run_pysr(x, y, w, DEFAULT_OPS, seed)
        pf = pareto_front_rows(d["front"])
        for row in pf:
            rows.append(
                {
                    "method": "pysr",
                    **row,
                    "selected": row["expression"] == d["selected"],
                }
            )
    save_table(pd.DataFrame(rows), AREA, "pareto")
    return {"rows": len(rows)}


def pareto_front_rows(front_rows: list[dict]) -> list[dict]:
    """PySR's table re-scored with our loss, reduced to a strict front, with
    the score recomputed on that loss."""
    ok = [r for r in front_rows if np.isfinite(r["loss"])]
    ok.sort(key=lambda r: r["complexity"])
    out: list[dict] = []
    for r in ok:
        if not out or r["loss"] < out[-1]["loss"]:
            out.append(dict(r))
    prev = None
    for r in out:
        r["score"] = (
            0.0
            if prev is None
            else (
                -float(np.log(r["loss"] / prev["loss"]))
                / (r["complexity"] - prev["complexity"])
                if r["loss"] > 0
                else float("inf")
            )
        )
        prev = r
    return out


# ---------------------------------------------------------------------------
# (d) SINDy
# ---------------------------------------------------------------------------

SINDY_NOISE = (0.0, 1e-3, 1e-2, 3e-2, 1e-1)
SINDY_THRESHOLDS = (0.05, 0.1, 0.2, 0.5, 1.0)
SINDY_WINDOWS = (11, 21, 51, 101)


def _sindy_runs(seeds, thresholds, windows, noises):
    from .sindy import (
        add_noise,
        damped_oscillator,
        fit_sindy,
        pendulum,
        score_model,
        simulate,
    )

    rows = []
    for sys in (damped_oscillator(), pendulum()):
        t, X = simulate(sys)
        for noise in noises:
            for seed in seeds:
                Xn = add_noise(X, noise, np.random.default_rng(seed))
                for th in thresholds:
                    for method in ("fd", "savgol"):
                        for win in windows if method == "savgol" else (0,):
                            m = fit_sindy(
                                t,
                                Xn,
                                sys.state,
                                threshold=th,
                                method=method,
                                window=win or 21,
                                trig=sys.trig,
                            )
                            rows.append(
                                {
                                    "system": sys.name,
                                    "noise": noise,
                                    "seed": seed,
                                    "threshold": th,
                                    "method": method,
                                    "window": win,
                                    **score_model(m, sys),
                                    "equations": " ; ".join(m.equations()),
                                }
                            )
    return pd.DataFrame(rows)


def study_sindy(quick: bool = False) -> dict:
    from .sindy import derivative_convergence, noise_amplification

    noises = SINDY_NOISE if not quick else SINDY_NOISE[:3]
    # tuning: threshold (and window) per system and derivative method, on
    # seeds 3/7/19, by support accuracy then coefficient error, averaged
    # over the noise levels
    tune = _sindy_runs(
        TUNE_SEEDS if not quick else TUNE_SEEDS[:1],
        SINDY_THRESHOLDS,
        SINDY_WINDOWS,
        noises,
    )
    agg = (
        tune.groupby(["system", "method", "threshold", "window"])
        .agg(support=("support_ok", "mean"), err=("coef_rel_err", "median"))
        .reset_index()
    )
    chosen = (
        agg.sort_values(["support", "err"], ascending=[False, True])
        .groupby(["system", "method"])
        .head(1)
    )
    save_table(agg, AREA, "sindy_tune")
    rows = []
    rep = _sindy_runs(
        REPORT_SEEDS if not quick else REPORT_SEEDS[:1],
        sorted(set(chosen["threshold"])),
        sorted(set(chosen["window"])),
        noises,
    )
    for _, c in chosen.iterrows():
        sel = rep[
            (rep.system == c.system)
            & (rep.method == c.method)
            & (rep.threshold == c.threshold)
            & (rep.window == c.window)
        ]
        rows.append(sel)
    df = pd.concat(rows, ignore_index=True)
    save_table(df, AREA, "sindy_recovery")
    save_json({"chosen": chosen.to_dict("records")}, AREA, "sindy_chosen")
    save_table(
        pd.DataFrame(derivative_convergence()), AREA, "sindy_derivative_convergence"
    )
    save_table(pd.DataFrame(noise_amplification()), AREA, "sindy_noise_amplification")
    return {"rows": len(df)}


# ---------------------------------------------------------------------------
# what the fitting packages actually do
# ---------------------------------------------------------------------------


def study_packages(quick: bool = False) -> dict:
    """Small, exact checks of the behaviour docs/theory/packages.md describes."""
    from scipy.optimize import curve_fit, least_squares

    out: dict = {}
    # 1. curve_fit covariance: coverage of the 1-sigma interval over many
    #    simulated datasets, with the stated sigma right and wrong by 2x
    rng = np.random.default_rng(REPORT_SEEDS[0])
    x = np.linspace(0.0, 2.0, 20)
    a0, b0, sig = 2.0, 1.3, 0.05

    def f(x, a, b):
        return a * np.exp(-b * x)

    n_sim = 2000 if not quick else 200
    cover = {}
    for stated in (1.0, 2.0):
        for absolute in (True, False):
            hits = 0
            for _ in range(n_sim):
                y = f(x, a0, b0) + sig * rng.standard_normal(x.size)
                p, c = curve_fit(
                    f,
                    x,
                    y,
                    p0=(1, 1),
                    sigma=np.full(x.size, stated * sig),
                    absolute_sigma=absolute,
                )
                hits += abs(p[1] - b0) <= np.sqrt(c[1, 1])
            cover[f"stated_{stated:g}x_absolute_{absolute}"] = hits / n_sim
    out["coverage_1sigma"] = cover
    out["coverage_n_sim"] = n_sim

    # 2. pcov is (J^T J)^-1 s^2: rebuild it by hand from the Jacobian
    y = f(x, a0, b0) + sig * np.random.default_rng(REPORT_SEEDS[1]).standard_normal(
        x.size
    )
    p_lm, c_lm = curve_fit(f, x, y, p0=(1, 1), method="lm")
    p_trf, c_trf = curve_fit(f, x, y, p0=(1, 1), method="trf")
    res = least_squares(lambda p: f(x, *p) - y, p_lm)
    J = res.jac
    s2 = float(np.sum(res.fun**2) / (x.size - 2))
    c_hand = np.linalg.inv(J.T @ J) * s2
    out["lm_vs_trf_param_rel_diff"] = float(np.max(np.abs(p_lm - p_trf) / np.abs(p_lm)))
    out["pcov_vs_hand_rel_diff"] = float(np.max(np.abs(c_lm - c_hand) / np.abs(c_hand)))
    out["lm_vs_trf_pcov_rel_diff"] = float(np.max(np.abs(c_lm - c_trf) / np.abs(c_lm)))

    # 3. autograd: a second derivative needs create_graph=True
    try:
        import torch

        t = torch.linspace(0.0, 3.0, 50, dtype=torch.float64, requires_grad=True)
        u = torch.sin(t)
        du = torch.autograd.grad(u.sum(), t)[0]
        try:
            torch.autograd.grad(du.sum(), t)
            out["second_derivative_without_create_graph"] = "no error"
        except RuntimeError as exc:
            out["second_derivative_without_create_graph"] = str(exc).split("\n")[0][
                :120
            ]
        u = torch.sin(t)
        du = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), t)[0].detach()
        out["second_derivative_max_err"] = float(
            (d2u + torch.sin(t)).abs().max().item()
        )

        # 4. Adam's first step is lr * sign(g), whatever the gradient's scale
        steps = {}
        for scale in (1e-6, 1.0, 1e6):
            p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
            opt = torch.optim.Adam([p], lr=1e-3)
            opt.zero_grad()
            (scale * p).sum().backward()
            opt.step()
            steps[f"{scale:g}"] = float(1.0 - p.item())
        out["adam_first_step"] = steps
        out["torch_version"] = torch.__version__
    except ImportError:
        out["torch"] = "not installed"
    import scipy
    import sympy

    out["scipy_version"] = scipy.__version__
    out["sympy_version"] = sympy.__version__
    from importlib.metadata import PackageNotFoundError, version

    # read from the metadata: importing pysr here would start Julia after
    # torch, the order the package __init__ exists to prevent
    try:
        out["pysr_version"] = version("pysr")
    except PackageNotFoundError:
        out["pysr_version"] = None
    save_json(out, AREA, "packages_checks")
    return out


# ---------------------------------------------------------------------------

PARTS = {
    "growth": study_growth,
    "tune": study_tune,
    "noise": study_noise,
    "budget": study_budget,
    "pysr": study_pysr,
    "vocabulary": study_vocabulary,
    "pareto": study_pareto,
    "sindy": study_sindy,
    "packages": study_packages,
}


def run(quick: bool = False, parts: list[str] | None = None, docs: bool = True) -> dict:
    """Run the named parts (all by default, in an order where `tune` precedes
    the GP studies), then redraw the figures and, if `docs`, re-render
    docs/theory/. Figures and docs read whatever results exist, so parts can
    also be run in separate processes and this called last with parts=[]."""
    from .figures import make_figures
    from .report import render_doc

    out: dict = {}
    for name in PARTS if parts is None else parts:
        t0 = time.time()
        print(f"[symbolic] {name} ...", flush=True)
        out[name] = PARTS[name](quick)
        print(f"[symbolic] {name} done in {time.time() - t0:.0f}s", flush=True)
    if parts is None or not parts:
        out["figures"] = [str(p) for p in make_figures()]
        # a quick run must not overwrite the committed pages
        if docs and not quick:
            out["docs"] = [str(p) for p in render_doc()]
    return out
