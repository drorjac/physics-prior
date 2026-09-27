"""The reconstruction study: how the value of a physics prior scales with the
dimension of the field.

    physprior.reconstruction.study.run(quick=False)

Work is split into independent units (case x dimension x seed, and so on)
and run on a small process pool, one BLAS/torch thread per worker. Every
random draw is keyed on (seed, dimension, case), so a unit gives the same
numbers whichever worker runs it.

Seed discipline: the MLP configuration and the interpolation smoothing are
chosen on the tuning seeds 3/7/19; every reported number comes from the
reporting seeds 11/23/42. The PINN's PDE weight is fixed in advance (the
residual is scaled by the peak of a unit source, so w_pde = 1e-2 weights a
source-sized residual like a 10% data misfit) and is not tuned: at the
machine load of this run a tuning sweep of the PINN would cost more than the
whole remaining study.
"""

from __future__ import annotations

import functools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import save_json, save_table
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS

from . import fields as F
from . import methods as M

TRACK = "reconstruction"
NOISE = 0.01  # sensor noise, in units of the field's standard deviation
TARGETS = (0.1, 0.03)
HEADLINE_TARGET = 0.1

N_GRID = {
    1: (4, 8, 16, 32, 64, 128),
    2: (8, 16, 32, 64, 128, 256, 512),
    3: (16, 32, 64, 128, 256, 512, 1024, 2048),
}
N_GRID_QUICK = {1: (4, 16), 2: (8, 32), 3: (16, 64)}
PHYSICS_N_MAX = 1024  # at the noise floor well before this; saves minutes

NOISE_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
NOISE_N = {1: 32, 2: 128, 3: 512}
NOISE_METHODS = ("physics", "gp", "interp")

PINN_N = {1: (8, 32), 2: (32, 128), 3: (128, 512)}
PINN_EPOCHS_STUDY = {1: 1500, 2: 1200, 3: 800}
PINN_W_PDE = 1e-2

NN_TUNE_GRID: tuple[dict[str, Any], ...] = (
    {"width": 32, "depth": 3, "weight_decay": 1e-4},
    {"width": 64, "depth": 4, "weight_decay": 1e-4},
    {"width": 32, "depth": 3, "weight_decay": 1e-2},
)
NN_EPOCHS = 2500
TUNE_N = {1: 32, 2: 128, 3: 512}

MISMATCH_KINDS = ("extra_source", "neumann_wall")
MISMATCH_N = (16, 32, 64, 128, 256, 512)
MISMATCH_METHODS = ("physics", "pigp", "gp", "interp")

MAIN_METHODS = ("physics", "pigp", "gp", "interp", "nn")


# ---------------------------------------------------------------------------
# scenes and scoring


@dataclass
class Scene:
    field: F.Field
    X: np.ndarray  # dense evaluation grid
    truth: np.ndarray
    scale: float  # std of the truth inside the sensor region
    in_region: np.ndarray  # eval points inside the sensor region


def make_scene(field: F.Field) -> Scene:
    X = F.eval_grid(field.geom)
    truth = field.value(X)
    lo, hi = field.geom.sensor_region
    inside = np.all((lo <= X) & (hi >= X), axis=1)
    return Scene(field, X, truth, float(np.std(truth[inside])), inside)


def _hull_mask(x: np.ndarray, X: np.ndarray) -> np.ndarray:
    if x.shape[1] == 1:
        return (X[:, 0] >= x[:, 0].min()) & (X[:, 0] <= x[:, 0].max())
    from scipy.spatial import Delaunay

    try:
        return Delaunay(x).find_simplex(X) >= 0
    except Exception:  # degenerate design
        return np.zeros(len(X), bool)


def _nrmse(err: np.ndarray, scale: float) -> float:
    if len(err) == 0:
        return float("nan")
    if not np.all(np.isfinite(err)):
        return float("inf")
    return float(np.sqrt(np.mean(err**2)) / scale)


def score(fit, sc: Scene, x: np.ndarray) -> dict[str, Any]:
    pred = fit.predict(sc.X)
    err = pred - sc.truth
    if sc.field.geom.case == "box":
        inner = _hull_mask(x, sc.X)
        main = _nrmse(err, sc.scale)
    else:
        inner = sc.in_region
        main = _nrmse(err[inner], sc.scale)
    loc = float("nan")
    if "centers" in fit.extra:
        loc = M.location_error(sc.field.centers, np.asarray(fit.extra["centers"]))
    at_bound = fit.extra.get("at_bound", False)
    if isinstance(at_bound, dict):
        at_bound = any(at_bound.values())
    return {
        "nrmse": main,
        "nrmse_in": _nrmse(err[inner], sc.scale),
        "nrmse_out": _nrmse(err[~inner], sc.scale),
        "frac_out": float(np.mean(~inner)),
        "loc_err": loc,
        "at_bound": bool(at_bound),
        "seconds": float(fit.seconds),
    }


def fit_method(name, x, y, geom, seed, tuned, *, phys=None):
    if name == "physics":
        return M.fit_physics(x, y, geom, seed=seed)
    if name == "pigp":
        phys = phys or M.fit_physics(x, y, geom, seed=seed)
        f = M.fit_gp(x, y, mean=phys.predict, name="pigp", seed=seed)
        f.seconds += phys.seconds
        return f
    if name == "gp":
        return M.fit_gp(x, y, seed=seed)
    if name == "interp":
        return M.fit_interp(x, y, smoothing=tuned["interp"][geom.case][str(geom.d)])
    if name == "nn":
        return M.fit_nn(
            x, y, cfg=tuned["nn"][str(geom.d)], seed=seed, epochs=tuned["nn_epochs"]
        )
    if name == "pinn":
        return M.fit_pinn(
            x,
            y,
            geom,
            w_pde=PINN_W_PDE,
            seed=seed,
            epochs=tuned.get("pinn_epochs", PINN_EPOCHS_STUDY)[geom.d],
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# work units (module-level so a process pool can pickle them)


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def unit_tune(d: int, quick: bool) -> dict[str, Any]:
    """MLP configuration and RBF smoothing, chosen on the tuning seeds."""
    n = N_GRID_QUICK[d][-1] if quick else TUNE_N[d]
    grid = NN_TUNE_GRID[:1] if quick else NN_TUNE_GRID
    epochs = 200 if quick else NN_EPOCHS
    nn_scores = np.zeros(len(grid))
    interp: dict[str, float] = {}
    for case in F.CASES:
        sm_scores = np.zeros(len(M.INTERP_SMOOTHING))
        for seed in TUNE_SEEDS:
            sc = make_scene(F.draw_field(case, d, seed))
            x, y, _ = F.sensors(sc.field, n, seed, NOISE, sc.scale)
            for i, s in enumerate(M.INTERP_SMOOTHING):
                sm_scores[i] += score(M.fit_interp(x, y, smoothing=s), sc, x)["nrmse"]
            if case == "box":
                for i, cfg in enumerate(grid):
                    f = M.fit_nn(x, y, cfg=cfg, seed=seed, epochs=epochs)
                    nn_scores[i] += score(f, sc, x)["nrmse"]
        interp[case] = float(M.INTERP_SMOOTHING[int(np.argmin(sm_scores))])
    k = int(np.argmin(nn_scores))
    return {
        "d": d,
        "nn": dict(grid[k]),
        "nn_scores": (nn_scores / len(TUNE_SEEDS)).tolist(),
        "interp": interp,
    }


def unit_main(case: str, d: int, seed: int, tuned: dict, quick: bool) -> list[dict]:
    sc = make_scene(F.draw_field(case, d, seed))
    grid = N_GRID_QUICK[d] if quick else N_GRID[d]
    x_all, y_all, _ = F.sensors(sc.field, max(grid), seed, NOISE, sc.scale)
    rows = []
    for n in grid:
        x, y = x_all[:n], y_all[:n]
        phys = None
        for name in MAIN_METHODS:
            if name in ("physics", "pigp") and n > PHYSICS_N_MAX:
                continue
            fit = fit_method(name, x, y, sc.field.geom, seed, tuned, phys=phys)
            if name == "physics":
                phys = fit
            rows.append(
                {"case": case, "d": d, "N": n, "seed": seed, "noise": NOISE}
                | {"method": name, "P": sc.field.geom.n_unknowns()}
                | score(fit, sc, x)
            )
    return rows


def unit_pinn(case: str, d: int, seed: int, tuned: dict, quick: bool) -> list[dict]:
    sc = make_scene(F.draw_field(case, d, seed))
    grid = (N_GRID_QUICK[d][0],) if quick else PINN_N[d]
    x_all, y_all, _ = F.sensors(sc.field, max(grid), seed, NOISE, sc.scale)
    rows = []
    for n in grid:
        x, y = x_all[:n], y_all[:n]
        fit = fit_method("pinn", x, y, sc.field.geom, seed, tuned)
        rows.append(
            {"case": case, "d": d, "N": n, "seed": seed, "noise": NOISE}
            | {"method": "pinn", "P": sc.field.geom.n_unknowns()}
            | score(fit, sc, x)
            | {"pinn_data_loss": fit.extra["final_data_loss"]}
        )
    return rows


def unit_noise(d: int, seed: int, tuned: dict, quick: bool) -> list[dict]:
    sc = make_scene(F.draw_field("box", d, seed))
    n = N_GRID_QUICK[d][-1] if quick else NOISE_N[d]
    levels = NOISE_GRID[::2] if quick else NOISE_GRID
    rows = []
    for lvl in levels:
        x, y, _ = F.sensors(sc.field, n, seed, lvl, sc.scale)
        for name in NOISE_METHODS:
            fit = fit_method(name, x, y, sc.field.geom, seed, tuned)
            rows.append(
                {"case": "box", "d": d, "N": n, "seed": seed, "noise": lvl}
                | {"method": name}
                | score(fit, sc, x)
            )
    return rows


def unit_mismatch(kind: str, seed: int, tuned: dict, quick: bool) -> list[dict]:
    field = F.mismatch_field(kind, seed)
    sc = make_scene(field)
    grid = MISMATCH_N[:2] if quick else MISMATCH_N
    x_all, y_all, _ = F.sensors(field, max(grid), seed, NOISE, sc.scale)
    rows = []
    for n in grid:
        x, y = x_all[:n], y_all[:n]
        phys = None
        for name in MISMATCH_METHODS:
            fit = fit_method(name, x, y, field.geom, seed, tuned, phys=phys)
            if name == "physics":
                phys = fit
            rows.append(
                {"kind": kind, "d": field.d, "N": n, "seed": seed}
                | {"method": name}
                | score(fit, sc, x)
            )
    return rows


# ---------------------------------------------------------------------------
# summaries


def samples_needed(
    df: pd.DataFrame, target: float, metric: str = "nrmse"
) -> pd.DataFrame:
    """N at which the median error over seeds first reaches `target`,
    log-interpolated between grid points; censored (a lower bound) if never."""
    rows = []
    for (case, d, method), g in df.groupby(["case", "d", "method"]):
        med = g.groupby("N")[metric].median().sort_index()
        ns, es = med.index.to_numpy(float), med.to_numpy()
        hit = np.flatnonzero(es <= target)
        if len(hit) == 0:
            n_star, censored = float(ns[-1]), True
        elif hit[0] == 0:
            n_star, censored = float(ns[0]), False
        else:
            i = hit[0]
            e0, e1 = np.log(es[i - 1]), np.log(es[i])
            t = (e0 - np.log(target)) / (e0 - e1) if e0 != e1 else 1.0
            n_star = float(np.exp(np.log(ns[i - 1]) + t * np.log(ns[i] / ns[i - 1])))
            censored = False
        rows.append(
            {
                "case": case,
                "d": int(d),
                "method": method,
                "metric": metric,
                "target": target,
                "N_star": n_star,
                "censored": censored,
                "at_first_grid_point": bool(len(hit) and hit[0] == 0),
                "N_max": float(ns[-1]),
                "P": int(F.Geometry(case, int(d)).n_unknowns()),
            }
        )
    return pd.DataFrame(rows)


def growth(ns: pd.DataFrame) -> pd.DataFrame:
    """Slope of log10 N* against d (decades per dimension)."""
    rows = []
    for (case, method, metric, target), g in ns.groupby(
        ["case", "method", "metric", "target"]
    ):
        g = g.sort_values("d")
        if len(g) < 2:
            continue
        slope = np.polyfit(g.d, np.log10(g.N_star), 1)[0]
        rows.append(
            {
                "case": case,
                "method": method,
                "metric": metric,
                "target": target,
                "decades_per_dim": float(slope),
                "ratio_3d_1d": float(
                    g[g.d == 3].N_star.iloc[0] / g[g.d == 1].N_star.iloc[0]
                )
                if {1, 3} <= set(g.d)
                else float("nan"),
                "any_censored": bool(g.censored.any()),
                "any_at_first_grid_point": bool(g.at_first_grid_point.any()),
            }
        )
    return pd.DataFrame(rows)


def _swap_nstar(nstar, d, m):
    return nstar(m, d)


def verdict(ns: pd.DataFrame, target: float = HEADLINE_TARGET) -> dict[str, Any]:
    """The three pre-registered criteria of docs/reconstruction/HYPOTHESIS.md."""
    out: dict[str, Any] = {"target": target}
    sub = ns[(ns.metric == "nrmse") & (ns.target == target)]
    for case in sorted(sub.case.unique()):
        s = sub[sub.case == case].set_index(["method", "d"])
        dims = sorted(s.index.get_level_values("d").unique())
        if not {1, 3} <= set(dims):
            continue

        def nstar(m, d, s=s):
            return float(s.loc[(m, d), "N_star"]), bool(s.loc[(m, d), "censored"])

        black = [m for m in ("nn", "gp", "interp") if (m, 1) in s.index]
        # ties (all censored at the budget) go to an uncensored method
        best: dict[int, str] = {}
        for d in dims:
            best[d] = min(black, key=functools.partial(_swap_nstar, nstar, d))
        b1, b1c = nstar(best[1], 1)
        b3, b3c = nstar(best[3], 3)
        p1, p1c = nstar("physics", 1)
        p3, p3c = nstar("physics", 3)
        P1 = F.Geometry(case, 1).n_unknowns()
        P3 = F.Geometry(case, 3).n_unknowns()
        c1 = b3 / b1 > 3.0
        c2 = p3 <= 3.0 * p1 * P3 / P1
        gap1, gap3 = b1 / p1, b3 / p3
        c3 = gap3 > gap1
        out[case] = {
            "best_black_box_by_d": {str(k): v for k, v in best.items()},
            "black_box_ratio_3d_1d": b3 / b1,
            "black_box_3d_censored": b3c,
            "physics_N_star": {"1": p1, "3": p3},
            "physics_censored": p1c or p3c,
            "physics_scaled_prediction_3d": p1 * P3 / P1,
            "gap_1d": gap1,
            "gap_3d": gap3,
            "criterion_1_black_box_grows": bool(c1),
            "criterion_2_physics_tracks_unknowns": bool(c2),
            "criterion_3_gap_widens": bool(c3),
            # a censored 1-D black box or physics value leaves a ratio undefined
            "determinate": not (b1c or p1c or p3c),
            "supported": bool(c1 and c2 and c3 and not (b1c or p1c or p3c)),
        }
    return out


# ---------------------------------------------------------------------------


def _pool_map(pool, fn, jobs):
    futs = [pool.submit(fn, *j) for j in jobs]
    out = []
    for f in futs:
        r = f.result()
        out.extend(r if isinstance(r, list) else [r])
    return out


def run(quick: bool = False, workers: int | None = None) -> dict[str, Any]:
    from physprior.logging import get_logger

    log = get_logger(__name__)
    t_start = time.time()
    workers = workers or (1 if quick else 3)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # workers are spawned and re-import the package; nothing here needs Julia
    os.environ.setdefault("PHYSPRIOR_NO_JULIA", "1")
    seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS
    dims = (1, 2, 3)
    timing: dict[str, float] = {}

    from . import convergence

    t0 = time.time()
    conv = convergence.run_all(quick=quick)
    for name, df in conv.items():
        save_table(df, TRACK, name)
    timing["convergence"] = time.time() - t0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        t0 = time.time()
        tunes = _pool_map(pool, unit_tune, [(d, quick) for d in dims])
        tuned: dict[str, Any] = {
            "nn": {str(t["d"]): t["nn"] for t in tunes},
            "nn_scores": {str(t["d"]): t["nn_scores"] for t in tunes},
            "interp": {
                c: {str(t["d"]): t["interp"][c] for t in tunes} for c in F.CASES
            },
            "nn_epochs": 200 if quick else NN_EPOCHS,
            "pinn_epochs": {1: 150, 2: 100, 3: 60} if quick else PINN_EPOCHS_STUDY,
            "pinn_w_pde": PINN_W_PDE,
            "tune_seeds": list(TUNE_SEEDS),
        }
        save_json(tuned, TRACK, "tuning")
        timing["tuning"] = time.time() - t0

        t0 = time.time()
        # heavy units first so the pool drains evenly
        jobs_main = [
            (c, d, s, tuned, quick) for d in (3, 2, 1) for c in F.CASES for s in seeds
        ]
        jobs_pinn = [("box", d, s, tuned, quick) for d in (3, 2, 1) for s in seeds]
        futs_main = [pool.submit(unit_main, *j) for j in jobs_main]
        futs_pinn = [pool.submit(unit_pinn, *j) for j in jobs_pinn]
        futs_noise = [
            pool.submit(unit_noise, d, s, tuned, quick) for d in dims for s in seeds
        ]
        futs_mis = [
            pool.submit(unit_mismatch, k, s, tuned, quick)
            for k in MISMATCH_KINDS
            for s in seeds
        ]
        main = [r for f in futs_main for r in f.result()]
        pinn = [r for f in futs_pinn for r in f.result()]
        noise = [r for f in futs_noise for r in f.result()]
        mis = [r for f in futs_mis for r in f.result()]
        timing["sweeps"] = time.time() - t0

    sweep = pd.DataFrame(main + pinn)
    save_table(sweep, TRACK, "sweep")
    save_table(pd.DataFrame(noise), TRACK, "noise")
    save_table(pd.DataFrame(mis), TRACK, "mismatch")

    ns = pd.concat(
        [
            samples_needed(sweep[sweep.method != "pinn"], t, m)
            for t in TARGETS
            for m in ("nrmse", "nrmse_out")
        ],
        ignore_index=True,
    )
    save_table(ns, TRACK, "samples_needed")
    gr = growth(ns)
    save_table(gr, TRACK, "growth")
    v = verdict(ns)
    save_json(v, TRACK, "verdict")

    from . import plots

    t0 = time.time()
    plots.make_all(quick=quick)
    timing["figures"] = time.time() - t0
    timing["total"] = time.time() - t_start

    meta = {
        "track": TRACK,
        "quick": quick,
        "seeds": list(seeds),
        "tune_seeds": list(TUNE_SEEDS),
        "noise": NOISE,
        "targets": list(TARGETS),
        "n_grid": {
            str(k): list(v) for k, v in (N_GRID_QUICK if quick else N_GRID).items()
        },
        "physics_n_max": PHYSICS_N_MAX,
        "source_width": F.SOURCE_WIDTH,
        "n_sources": F.N_SOURCES,
        "kmax_fit": F.KMAX_FIT,
        "kmax_truth": F.KMAX_TRUTH,
        "workers": workers,
        "threads_per_worker": 1,
        "timing_seconds": timing,
        "verdict": v,
    }
    save_json(meta, TRACK, "meta")
    from .doc import render_doc

    # a quick run must not overwrite the committed page
    render_doc(get_settings().results(TRACK) / "README.md" if quick else None)
    log.info("reconstruction study done in %.0f s", timing["total"])
    return meta
