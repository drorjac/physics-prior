"""The dynamics study: what physical structure buys a learned update rule.

Experiments
-----------
tune       learning rate per arm, on the tuning seeds and the validation set
           (pendulum for the ODE arms, Burgers for the PDE arms), then frozen
main       every arm on every system, one-step loss, reporting seeds
objective  one-step vs 5-step unrolled loss, same arms and gradient steps
data       number of training trajectories (pendulum, Burgers)

Every run trains for the same number of gradient steps with the same batch
size; wall time is recorded next to each result. Runs go to a process pool
with one torch thread per worker.

`run(quick)` writes results/dynamics/*.csv|json and figures/dynamics/*.png.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import save_json, save_table
from physprior.logging import get_logger
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS
from physprior.units import require

from . import TRACK
from . import metrics as M
from . import pdes as PD
from . import systems as S

log = get_logger(__name__)

ODE_SYSTEMS = ("pendulum", "duffing", "kepler", "lorenz")
PDE_SYSTEMS = ("heat", "advection", "burgers")
PDE_ARMS = ("conv", "dense")
BLACK_BOX = ("direct", "residual", "node")
ARM_ORDER = (
    "direct",
    "residual",
    "node",
    "hnn",
    "hnn_leapfrog",
    "closure",
    "conv",
    "dense",
    "oracle",
    "known_only",
    "physics",
)


@dataclass(frozen=True)
class StudyOptions:
    steps: int = 1000
    batch: int = 128
    pde_batch: int = 32  # one PDE sample is a whole 64-point field
    horizon: int = 5  # unroll length of the rollout loss
    n_traj: int = 32
    n_test: int = 16
    n_val: int = 8
    seeds: tuple[int, ...] = REPORT_SEEDS
    tune_seeds: tuple[int, ...] = TUNE_SEEDS
    # Widened from (1e-3, 3e-3) after the tuning-seed edge check put every arm
    # at the upper edge; see results/dynamics/tuning_edge_grid1.json.
    lrs: tuple[float, ...] = (3e-3, 1e-2, 3e-2)
    default_lr: float = 3e-3
    tune: bool = True
    data_sizes: tuple[int, ...] = (2, 8)
    objective_ode: tuple[str, ...] = ("pendulum", "lorenz")
    objective_pde: tuple[str, ...] = ("burgers",)
    ode_systems: tuple[str, ...] = ODE_SYSTEMS
    pde_systems: tuple[str, ...] = PDE_SYSTEMS
    horizon_cap: int | None = None  # truncate test rollouts (quick mode)
    workers: int = 3

    @classmethod
    def quick(cls) -> StudyOptions:
        return cls(
            steps=30,
            batch=32,
            pde_batch=8,
            horizon=3,
            n_traj=3,
            n_test=3,
            n_val=2,
            seeds=(11,),
            tune=False,
            data_sizes=(2,),
            objective_ode=("pendulum",),
            objective_pde=("burgers",),
            horizon_cap=60,
            workers=1,
        )


@dataclass(frozen=True)
class Task:
    kind: str  # "ode" | "pde"
    exp: str  # "tune" | "main" | "objective" | "data" | "ref"
    system: str
    arm: str
    seed: int
    lr: float
    n_traj: int
    loss: str = "onestep"
    examples: bool = False
    opts: StudyOptions = field(default_factory=StudyOptions, repr=False)


# ---------------------------------------------------------------------------
# data, cached on disk so the workers do not regenerate it


def _cache_dir() -> Path:
    p = get_settings().cache_dir / "dynamics"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cached(key: str, make: Callable[[], dict]) -> dict:
    path = _cache_dir() / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    arrs = {k: v for k, v in make().items() if v is not None}
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrs)
    tmp.replace(path)
    return arrs


@cache
def _system_hash(kind: str, name: str) -> str:
    """Part of every trajectory cache key: the system's parameters (dt,
    substeps, rollout lengths) and the source of the module that defines its
    field, initial-condition ranges and reference solver. Changing any of
    them makes a new key, so a stale trajectory file is never reused."""
    mod = S if kind == "ode" else PD
    sys = S.get_system(name) if kind == "ode" else PD.PDES[name]
    h = hashlib.sha256(repr(sys).encode())
    h.update(Path(inspect.getfile(mod)).read_bytes())
    return h.hexdigest()[:12]


def _n_roll(kind: str, name: str, opts: StudyOptions) -> int:
    n = S.get_system(name).test_steps if kind == "ode" else PD.PDES[name].test_steps
    return min(n, opts.horizon_cap) if opts.horizon_cap else n


def ode_data(name: str, n_traj: int, seed: int, opts: StudyOptions) -> S.OdeData:
    sys = S.get_system(name)
    n = _n_roll("ode", name, opts)
    tag = _system_hash("ode", name)
    fx = _cached(
        f"ode_{name}_fixed_{opts.n_test}_{opts.n_val}_{n}_{tag}",
        lambda: S.fixed_sets(sys, opts.n_test, opts.n_val, n),
    )
    fx.setdefault("ood", None)
    tr = _cached(
        f"ode_{name}_train_{n_traj}_{seed}_{tag}",
        lambda: {"train": S.train_set(sys, n_traj, seed)},
    )
    return S.assemble(sys, tr["train"], fx)


def pde_data(name: str, n_traj: int, seed: int, opts: StudyOptions) -> PD.PdeData:
    n = _n_roll("pde", name, opts)
    tag = _system_hash("pde", name)
    fx = _cached(
        f"pde_{name}_fixed_{opts.n_test}_{opts.n_val}_{n}_{tag}",
        lambda: PD.pde_fixed_sets(name, opts.n_test, opts.n_val, n),
    )
    tr = _cached(
        f"pde_{name}_train_{n_traj}_{seed}_{tag}",
        lambda: {"train": PD.pde_train_set(name, n_traj, seed)},
    )
    return PD.pde_assemble(name, tr["train"], fx)


# ---------------------------------------------------------------------------
# one run


def _build(task: Task, data):
    from . import steppers as ST

    if task.kind == "ode":
        return ST.build_ode(task.arm, S.get_system(task.system), data)
    sys = PD.PDES[task.system]
    a = (PD.N_GRID, sys.dt, data.mean, data.std, data.dstd)
    if task.arm == "conv":
        return ST.ConvStepper(*a)
    if task.arm == "dense":
        return ST.DenseStepper(*a)
    if task.arm == "closure":
        return ST.PdeClosure(*a, sys.known_rhs, data.rscale)
    if task.arm == "physics":
        return ST.PdeFixed(sys.dt, PD.physics_step(task.system))
    raise KeyError(task.arm)


REFERENCE = ("oracle", "known_only", "physics")


def run_task(task: Task) -> dict:
    from . import steppers as ST
    from . import train as TR

    o = task.opts
    data = (ode_data if task.kind == "ode" else pde_data)(
        task.system, task.n_traj, task.seed, o
    )
    model = _build(task, data)
    fit = {"seconds": 0.0, "final_loss": float("nan")}
    if task.arm not in REFERENCE:
        topts = TR.TrainOptions(
            steps=o.steps,
            batch=o.batch if task.kind == "ode" else o.pde_batch,
            lr=task.lr,
            loss=task.loss,
            horizon=o.horizon,
        )
        fit = TR.train(model, data.train, data.std, topts, task.seed)
    base = {
        "exp": task.exp,
        "system": task.system,
        "arm": task.arm,
        "seed": task.seed,
        "lr": task.lr,
        "loss": task.loss,
        "n_traj": task.n_traj,
        "train_seconds": fit["seconds"],
        "final_loss": fit["final_loss"],
        "n_params": ST.n_params(model),
    }
    splits = ("val",) if task.exp == "tune" else ("test", "ood")
    ev = evaluate_ode if task.kind == "ode" else evaluate_pde
    rows, curves, examples = [], [], {}
    for split in splits:
        truth = getattr(data, split)
        if truth is None:
            continue
        t0 = time.time()
        row, curve, ex = ev(model, task.system, data, truth, task.examples)
        row.update(base, split=split, eval_seconds=time.time() - t0)
        rows.append(row)
        for c in curve:
            c.update(
                {k: base[k] for k in ("exp", "system", "arm", "seed", "loss", "n_traj")}
            )
            c["split"] = split
        curves += curve
        if ex is not None:
            examples[split] = ex
    return {"rows": rows, "curves": curves, "examples": examples, "task": task}


def evaluate_ode(model, name: str, data, truth: np.ndarray, keep: bool):
    sys = S.get_system(name)
    n = truth.shape[1] - 1
    pred = M.rollout(model, truth[:, 0], n, data.mean, data.std)
    curve = M.error_curve(pred, truth, data.std)
    thr = M.LORENZ_THR if sys.chaotic else M.VALID_THR
    vs = M.valid_steps(curve, thr)
    h = min(sys.train_steps, n)
    row = {
        "onestep": M.onestep_error(model, truth, data.std),
        "err_train_h": float(np.median(curve[:, h])),
        "err_end": float(np.median(curve[:, -1])),
        "valid_steps": float(np.median(vs)),
        "valid_time": float(np.median(vs)) * sys.dt,
        "valid_over_train": float(np.median(vs)) / sys.train_steps,
        "blowup_frac": float(np.mean(~np.isfinite(pred).all(axis=(1, 2)))),
        # A median hides a minority of rollouts that go unstable; this does not.
        "diverged_frac": float(np.mean(curve[:, -1] > 1.0)),
        "horizon_steps": n,
    }
    if sys.chaotic:
        row["vpt_lyap"] = row["valid_time"] * S.LORENZ_LAMBDA1
        row["vpt_lyap_mean"] = float(np.mean(vs)) * sys.dt * S.LORENZ_LAMBDA1
    idx = M.log_indices(n)
    cur = {"step": idx, "t": idx * sys.dt, "err": np.median(curve[:, idx], axis=0)}
    for key, fn in sys.invariants.items():
        d = M.invariant_drift(pred, fn, key)
        early = M.window_mean(d, 0.1, last=False)
        late = M.window_mean(d, 0.1, last=True)
        row[f"{key}_early"] = float(np.median(early))
        row[f"{key}_late"] = float(np.median(late))
        with np.errstate(invalid="ignore", divide="ignore"):
            row[f"{key}_growth"] = float(np.median(late / early))
        cur[f"drift_{key}"] = np.median(d[:, idx], axis=0)
    curves = pd.DataFrame(cur).to_dict("records")
    ex = None
    if keep:
        ex = {"pred": pred[:4], "truth": truth[:4]}
        for key, fn in sys.invariants.items():
            ex[f"drift_{key}"] = np.median(M.invariant_drift(pred, fn, key), axis=0)
    return row, curves, ex


def evaluate_pde(model, name: str, data, truth: np.ndarray, keep: bool):
    sys = PD.PDES[name]
    n = truth.shape[1] - 1
    pred = M.rollout(model, truth[:, 0], n, data.mean, data.std)
    rms0 = truth[:, :1].std(axis=-1, keepdims=True)
    curve = M.error_curve(pred, truth, rms0)
    vs = M.valid_steps(curve, M.VALID_THR)
    h = min(sys.train_steps, n)
    with np.errstate(invalid="ignore", over="ignore"):
        mass = np.abs(pred[:, -1].mean(-1) - truth[:, 0].mean(-1)) / rms0[:, 0, 0]
        eratio = pred[:, -1].var(-1) / truth[:, -1].var(-1)
    row = {
        "onestep": M.onestep_error(model, truth, rms0),
        "err_train_h": float(np.median(curve[:, h])),
        "err_end": float(np.median(curve[:, -1])),
        "valid_steps": float(np.median(vs)),
        "valid_time": float(np.median(vs)) * sys.dt,
        "valid_over_train": float(np.median(vs)) / sys.train_steps,
        "blowup_frac": float(np.mean(~np.isfinite(pred).all(axis=(1, 2)))),
        # A median hides a minority of rollouts that go unstable; this does not.
        "diverged_frac": float(np.mean(curve[:, -1] > 1.0)),
        "mass_drift_end": float(np.median(np.where(np.isfinite(mass), mass, np.inf))),
        "energy_ratio_end": float(
            np.median(np.where(np.isfinite(eratio), eratio, np.inf))
        ),
        "horizon_steps": n,
    }
    idx = M.log_indices(n, 40)
    curves = pd.DataFrame(
        {"step": idx, "t": idx * sys.dt, "err": np.median(curve[:, idx], axis=0)}
    ).to_dict("records")
    ex = {"pred": pred[:2], "truth": truth[:2]} if keep else None
    return row, curves, ex


# ---------------------------------------------------------------------------
# task lists


def _ode_arms(name: str) -> list[str]:
    from .steppers import arms_for

    return arms_for(S.get_system(name))


def _pde_arms(name: str) -> list[str]:
    return list(PDE_ARMS) + (["closure"] if PD.PDES[name].known_rhs is not None else [])


def tune_tasks(o: StudyOptions) -> list[Task]:
    out = []
    for seed in o.tune_seeds:
        for lr in o.lrs:
            out += [
                Task("ode", "tune", "pendulum", a, seed, lr, o.n_traj, opts=o)
                for a in _ode_arms("pendulum")
            ]
            out += [
                Task("pde", "tune", "burgers", a, seed, lr, o.n_traj, opts=o)
                for a in _pde_arms("burgers")
            ]
    return out


def choose_lr(df: pd.DataFrame) -> dict[str, float]:
    """Per arm: the learning rate with the longest median validation horizon
    averaged over tuning seeds; ties go to the lower one-step error."""
    g = df.groupby(["arm", "lr"]).agg(v=("valid_steps", "mean"), e=("onestep", "mean"))
    g = g.reset_index().sort_values(["arm", "v", "e"], ascending=[True, False, True])
    return {a: float(sub.iloc[0]["lr"]) for a, sub in g.groupby("arm")}


def edge_check(o: StudyOptions, edge_lr: float = 1e-1) -> dict:
    """The tuned learning rate sits at the top of the grid for most arms.
    This re-runs the tuning tasks at one larger rate, on the tuning seeds and
    the validation set only, and records whether any arm would have chosen
    it. It does not change the frozen choices of the reported runs."""
    from physprior.io import load_table

    base = load_table(TRACK, "tuning")
    extra, _ = _frames(execute(tune_tasks(replace(o, lrs=(edge_lr,))), o.workers))
    both = pd.concat([base, extra], ignore_index=True)
    save_table(extra, TRACK, "tuning_edge")
    out: dict[str, Any] = {"edge_lr": edge_lr, "arms": {}}
    for system, prefix in (("pendulum", ""), ("burgers", "pde_")):
        sub = both[both.system == system]
        old = choose_lr(base[base.system == system])
        new = choose_lr(sub)
        for arm in old:
            g = sub[sub.arm == arm].groupby("lr").valid_steps.mean()
            out["arms"][prefix + arm] = {
                "chosen": old[arm],
                "with_edge": new[arm],
                "would_change": new[arm] != old[arm],
                "val_valid_steps_by_lr": {str(k): float(v) for k, v in g.items()},
            }
    save_json(out, TRACK, "tuning_edge")
    return out


def main_tasks(o: StudyOptions, lr: dict[str, float]) -> list[Task]:
    out = []
    for name in o.ode_systems:
        sysm = S.get_system(name)
        refs = ["oracle"] + (["known_only"] if sysm.known is not None else [])
        out += [
            Task("ode", "ref", name, a, o.seeds[0], 0.0, o.n_traj, opts=o) for a in refs
        ]
        for seed in o.seeds:
            out += [
                Task(
                    "ode",
                    "main",
                    name,
                    a,
                    seed,
                    lr.get(a, o.default_lr),
                    o.n_traj,
                    examples=seed == o.seeds[0],
                    opts=o,
                )
                for a in _ode_arms(name)
            ]
    for name in o.pde_systems:
        out.append(
            Task(
                "pde",
                "ref",
                name,
                "physics",
                o.seeds[0],
                0.0,
                o.n_traj,
                examples=True,
                opts=o,
            )
        )
        for seed in o.seeds:
            out += [
                Task(
                    "pde",
                    "main",
                    name,
                    a,
                    seed,
                    lr.get(f"pde_{a}", o.default_lr),
                    o.n_traj,
                    examples=seed == o.seeds[0],
                    opts=o,
                )
                for a in _pde_arms(name)
            ]
    return out


def objective_tasks(o: StudyOptions, lr: dict[str, float]) -> list[Task]:
    out = []
    for seed in o.seeds:
        for name in o.objective_ode:
            out += [
                Task(
                    "ode",
                    "objective",
                    name,
                    a,
                    seed,
                    lr.get(a, o.default_lr),
                    o.n_traj,
                    loss="rollout",
                    opts=o,
                )
                for a in BLACK_BOX
            ]
        for name in o.objective_pde:
            out += [
                Task(
                    "pde",
                    "objective",
                    name,
                    a,
                    seed,
                    lr.get(f"pde_{a}", o.default_lr),
                    o.n_traj,
                    loss="rollout",
                    opts=o,
                )
                for a in PDE_ARMS
            ]
    return out


def data_tasks(o: StudyOptions, lr: dict[str, float]) -> list[Task]:
    out = []
    for seed in o.seeds:
        for n in o.data_sizes:
            out += [
                Task(
                    "ode",
                    "data",
                    "pendulum",
                    a,
                    seed,
                    lr.get(a, o.default_lr),
                    n,
                    opts=o,
                )
                for a in _ode_arms("pendulum")
            ]
            out += [
                Task(
                    "pde",
                    "data",
                    "burgers",
                    a,
                    seed,
                    lr.get(f"pde_{a}", o.default_lr),
                    n,
                    opts=o,
                )
                for a in _pde_arms("burgers")
            ]
    return out


# ---------------------------------------------------------------------------
# execution


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


# Rough relative cost, so the longest tasks start first.
_COST = {"hnn": 5, "hnn_leapfrog": 4, "node": 4, "closure": 4, "dense": 2, "conv": 2}


def _cost(t: Task) -> float:
    c = _COST.get(t.arm, 1) * (4 if t.loss == "rollout" else 1)
    return c * (2 if t.system in ("kepler", "lorenz") else 1)


def execute(tasks: list[Task], workers: int) -> list[dict]:
    tasks = sorted(tasks, key=_cost, reverse=True)
    if workers <= 1:
        _init_worker()
        return [run_task(t) for t in tasks]
    import multiprocessing as mp

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
    ) as ex:
        return list(ex.map(run_task, tasks))


def _make_one(t: Task) -> None:
    (ode_data if t.kind == "ode" else pde_data)(t.system, t.n_traj, t.seed, t.opts)


def _prepare_data(tasks: list[Task], workers: int) -> None:
    """Generate every data set once, before training, so workers only load.
    The seed-independent sets go first so no two workers build the same one."""
    first: dict = {}
    rest: dict = {}
    for t in tasks:
        first.setdefault((t.kind, t.system), t)
        rest.setdefault((t.kind, t.system, t.n_traj, t.seed), t)
    for batch in (list(first.values()), list(rest.values())):
        if workers <= 1:
            for t in batch:
                _make_one(t)
            continue
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as ex:
            list(ex.map(_make_one, batch))


def _frames(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [r for res in results for r in res["rows"]]
    curves = [c for res in results for c in res["curves"]]
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---------------------------------------------------------------------------
# reference checks


def reference_checks(quick: bool) -> dict:
    out: dict = {"ode": [], "burgers": None, "lyapunov": None}
    for name in ODE_SYSTEMS:
        c = S.convergence(S.get_system(name), n_ics=2 if quick else 4)
        require(
            c["used_max_err"] < 1e-6,
            f"{name}: reference integrator error {c['used_max_err']:.2e} too large",
        )
        out["ode"].append(c)
    if not quick:
        b = PD.burgers_convergence()
        require(b["used_max_err"] < 1e-3, "Burgers reference not converged")
        out["burgers"] = b
        lam = S.lyapunov_exponent(200.0)
        out["lyapunov"] = {
            "measured": lam,
            "literature": S.LORENZ_LAMBDA1,
            "rel_diff": abs(lam - S.LORENZ_LAMBDA1) / S.LORENZ_LAMBDA1,
        }
    return out


# ---------------------------------------------------------------------------


def run(quick: bool = False, opts: StudyOptions | None = None) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PHYSPRIOR_NO_JULIA", "1")
    o = opts or (StudyOptions.quick() if quick else StudyOptions())
    t_start = time.time()
    checks = reference_checks(quick)
    save_json(checks, TRACK, "convergence")

    lr: dict[str, float] = {}
    tune_df = pd.DataFrame()
    if o.tune:
        tt = tune_tasks(o)
        _prepare_data(tt, o.workers)
        tune_df, _ = _frames(execute(tt, o.workers))
        save_table(tune_df, TRACK, "tuning")
        ode_lr = choose_lr(tune_df[tune_df.system == "pendulum"])
        pde_lr = choose_lr(tune_df[tune_df.system == "burgers"])
        lr = {**ode_lr, **{f"pde_{k}": v for k, v in pde_lr.items()}}
    at_edge = sorted(k for k, v in lr.items() if v in (min(o.lrs), max(o.lrs)))
    save_json(
        {"lr": lr, "tuned": o.tune, "grid": list(o.lrs), "at_grid_edge": at_edge},
        TRACK,
        "tuned_lr",
    )
    log.info("dynamics: tuning done in %.0f s, lr=%s", time.time() - t_start, lr)

    tasks = main_tasks(o, lr) + objective_tasks(o, lr) + data_tasks(o, lr)
    _prepare_data(tasks, o.workers)
    results = execute(tasks, o.workers)
    rows, curves = _frames(results)
    examples = {
        (r["task"].kind, r["task"].system, r["task"].arm): r["examples"]
        for r in results
        if r["examples"]
    }

    ode = rows[rows.system.isin(ODE_SYSTEMS)]
    pde = rows[rows.system.isin(PDE_SYSTEMS)]
    save_table(ode[ode.exp.isin(["main", "ref"])], TRACK, "ode_main")
    save_table(pde[pde.exp.isin(["main", "ref"])], TRACK, "pde_main")
    obj = rows[
        (rows.exp == "objective") | ((rows.exp == "main") & _in_objective(rows, o))
    ]
    save_table(obj, TRACK, "objective")
    dat = rows[
        (rows.exp == "data")
        | ((rows.exp == "main") & rows.system.isin(["pendulum", "burgers"]))
    ]
    save_table(dat, TRACK, "data_efficiency")
    save_table(curves, TRACK, "curves")
    _save_examples(examples)

    verdicts = evaluate_expectations(rows)
    save_json(verdicts, TRACK, "expectations")
    meta = {
        "options": asdict(o),
        "quick": quick,
        "seconds": time.time() - t_start,
        "n_runs": len(results) + len(tune_df),
        "lr": lr,
        "thresholds": {
            "valid": M.VALID_THR,
            "lorenz": M.LORENZ_THR,
            "blowup_z": M.BLOWUP_Z,
        },
    }
    save_json(meta, TRACK, "meta")

    from .figures import make_figures

    figs = make_figures(rows, curves, examples)
    log.info("dynamics: done in %.0f s", time.time() - t_start)
    return {"meta": meta, "expectations": verdicts, "figures": [str(f) for f in figs]}


def _in_objective(rows: pd.DataFrame, o: StudyOptions) -> pd.Series:
    ode = rows.system.isin(o.objective_ode) & rows.arm.isin(BLACK_BOX)
    pde = rows.system.isin(o.objective_pde) & rows.arm.isin(PDE_ARMS)
    return ode | pde


def _save_examples(examples: dict) -> None:
    """Seed-11 example rollouts, for the figures.

    Only the first rollout of each batch is kept, in float32: the figures
    draw that one, and the full batches in float64 made this the largest file
    in results/ by an order of magnitude. Every reported metric is computed
    before this point, from the full-precision rollouts.
    """
    arrs: dict[str, Any] = {}
    for (kind, system, arm), ex in examples.items():
        for split, d in ex.items():
            for k, v in d.items():
                a = np.asarray(v)
                arrs[f"{kind}__{system}__{arm}__{split}__{k}"] = (
                    a[:1] if a.ndim == 3 else a
                ).astype(np.float32)
    path = get_settings().results(TRACK) / "examples.npz"
    np.savez_compressed(path, **arrs)


def load_rows() -> pd.DataFrame:
    """All per-run rows of the last study, from results/dynamics."""
    from physprior.io import load_table

    obj = load_table(TRACK, "objective")
    dat = load_table(TRACK, "data_efficiency")
    return pd.concat(
        [
            load_table(TRACK, "ode_main"),
            load_table(TRACK, "pde_main"),
            obj[obj.exp == "objective"],
            dat[dat.exp == "data"],
        ],
        ignore_index=True,
    )


def load_examples() -> dict:
    path = get_settings().results_dir / TRACK / "examples.npz"
    out: dict = {}
    with np.load(path) as z:
        for key in z.files:
            kind, system, arm, split, k = key.split("__")
            out.setdefault((kind, system, arm), {}).setdefault(split, {})[k] = z[key]
    return out


# ---------------------------------------------------------------------------
# expectations: the criteria written down before the reported runs


def _med(
    rows: pd.DataFrame,
    system: str,
    arm: str,
    col: str,
    split: str = "test",
    exp: str = "main",
    **kw,
) -> float:
    m = rows[
        (rows.system == system)
        & (rows.arm == arm)
        & (rows.split == split)
        & (rows.exp == exp)
    ]
    for k, v in kw.items():
        m = m[m[k] == v]
    return float(np.median(m[col])) if len(m) else float("nan")


def _verdict(ok: bool | None) -> str:
    return "not evaluated" if ok is None else ("supported" if ok else "refuted")


def evaluate_expectations(rows: pd.DataFrame) -> dict:
    out: dict = {}
    det: dict[Any, Any] = {}
    has = set(rows.system)

    # E1 / E2: energy on the conservative systems
    cons = [(s, k) for s, k in (("pendulum", "H"), ("kepler", "E")) if s in has]
    e1: list[bool] = []
    e2: list[bool] = []
    det1: dict[str, Any] = {}
    det2: dict[str, Any] = {}
    for s, k in cons:
        lf_late = _med(rows, s, "hnn_leapfrog", f"{k}_late")
        lf_g = _med(rows, s, "hnn_leapfrog", f"{k}_growth")
        bbl = {a: _med(rows, s, a, f"{k}_late") for a in BLACK_BOX}
        e1.append(lf_late < min(bbl.values()) and lf_g <= 10)
        det1[s] = {
            "leapfrog_late": lf_late,
            "leapfrog_growth": lf_g,
            "black_box_late": bbl,
        }
        hg = _med(rows, s, "hnn", f"{k}_growth")
        e2.append(hg > lf_g)
        det2[s] = {
            "hnn_rk4_growth": hg,
            "leapfrog_growth": lf_g,
            "hnn_rk4_late": _med(rows, s, "hnn", f"{k}_late"),
        }
    out["E1"] = {"verdict": _verdict(all(e1) if e1 else None), "detail": det1}
    out["E2"] = {"verdict": _verdict(any(e2) if e2 else None), "detail": det2}

    # E3: residual vs direct
    odes = [s for s in ODE_SYSTEMS if s in has]
    det = {}
    wins_id: list[bool]
    wins_ood: list[bool]
    wins_id, wins_ood = [], []
    for s in odes:
        d, r = (
            _med(rows, s, "direct", "valid_steps"),
            _med(rows, s, "residual", "valid_steps"),
        )
        det[s] = {"direct": d, "residual": r}
        wins_id.append(d >= r)
        if not S.get_system(s).chaotic:
            do = _med(rows, s, "direct", "valid_steps", "ood")
            ro = _med(rows, s, "residual", "valid_steps", "ood")
            det[s].update(direct_ood=do, residual_ood=ro)
            wins_ood.append(do >= ro)
    ok = None
    if odes:
        ok = not (sum(wins_id) > len(wins_id) / 2 or sum(wins_ood) > len(wins_ood) / 2)
    out["E3"] = {"verdict": _verdict(ok), "detail": det}

    # E4: neural ODE vs residual
    det, w = {}, []
    for s in odes:
        n_, r = (
            _med(rows, s, "node", "valid_steps"),
            _med(rows, s, "residual", "valid_steps"),
        )
        det[s] = {"node": n_, "residual": r}
        w.append(n_ > r)
    out["E4"] = {"verdict": _verdict(sum(w) > len(w) / 2 if w else None), "detail": det}

    # E5: closure
    det, w = {}, []
    for s in ("pendulum", "duffing", "lorenz", "burgers"):
        if s not in has:
            continue
        bb: tuple[str, ...] = BLACK_BOX if s in ODE_SYSTEMS else PDE_ARMS
        if s == "lorenz":
            c = _med(rows, s, "closure", "valid_steps")
            b = {a: _med(rows, s, a, "valid_steps") for a in bb}
            ok_s = c > max(b.values())
        else:
            c = _med(rows, s, "closure", "err_train_h", "ood")
            b = {a: _med(rows, s, a, "err_train_h", "ood") for a in bb}
            ok_s = c < min(b.values())
        det[s] = {"closure": c, "black_box": b, "closure_best": bool(ok_s)}
        w.append(ok_s)
    out["E5"] = {"verdict": _verdict(sum(w) > len(w) / 2 if w else None), "detail": det}

    # E6: unrolled loss
    det, w = {}, []
    ob = rows[rows.exp == "objective"]
    for (s, a), _ in ob.groupby(["system", "arm"]):
        one = _med(rows, s, a, "valid_steps")
        unr = _med(rows, s, a, "valid_steps", exp="objective")
        det[f"{s}/{a}"] = {"onestep_loss": one, "rollout_loss": unr}
        w.append(unr > one)
    out["E6"] = {"verdict": _verdict(sum(w) > len(w) / 2 if w else None), "detail": det}

    # E7: conv vs dense
    det, w = {}, []
    for s in PDE_SYSTEMS:
        if s not in has:
            continue
        c, d = _med(rows, s, "conv", "err_end"), _med(rows, s, "dense", "err_end")
        co, do = (
            _med(rows, s, "conv", "err_end", "ood"),
            _med(rows, s, "dense", "err_end", "ood"),
        )
        det[s] = {"conv": c, "dense": d, "conv_ood": co, "dense_ood": do}
        w.append(d <= c)
    ratio: dict[int, float] = {}
    for n in sorted(set(rows[(rows.system == "burgers")].n_traj)):
        exp = "main" if n == rows[rows.exp == "main"].n_traj.max() else "data"
        c = _med(rows, "burgers", "conv", "err_end", exp=exp, n_traj=n)
        d = _med(rows, "burgers", "dense", "err_end", exp=exp, n_traj=n)
        ratio[int(n)] = c / d if d else float("nan")
    det["burgers_conv_over_dense_by_n_traj"] = ratio
    ok = None
    if w:
        ns = sorted(ratio)
        trend = len(ns) < 2 or ratio[ns[0]] < ratio[ns[-1]]
        ok = not (sum(w) > len(w) / 2) and trend
    out["E7"] = {"verdict": _verdict(ok), "detail": det}

    # E8: Lorenz, one-step error vs valid time
    lz = rows[(rows.system == "lorenz") & (rows.exp == "main") & (rows.split == "test")]
    if len(lz) >= 3:
        rho = float(
            pd.Series(lz.onestep.values)
            .rank()
            .corr(pd.Series(lz.vpt_lyap.values).rank())
        )
        out["E8"] = {
            "verdict": _verdict(rho <= -0.5),
            "detail": {
                "spearman": rho,
                "vpt_lyap_by_arm": lz.groupby("arm").vpt_lyap.median().to_dict(),
                "max_vpt_lyap": float(lz.vpt_lyap.max()),
            },
        }
    else:
        out["E8"] = {"verdict": _verdict(None), "detail": {}}

    # E9: data efficiency, pendulum
    pe = rows[
        (rows.system == "pendulum")
        & (rows.split == "test")
        & rows.exp.isin(["main", "data"])
    ]
    det, ok = {}, None
    ns = sorted(set(pe.n_traj))
    if len(ns) >= 2:
        for n in ns:
            r = float(
                np.median(pe[(pe.n_traj == n) & (pe.arm == "residual")].valid_steps)
            )
            lf = float(
                np.median(pe[(pe.n_traj == n) & (pe.arm == "hnn_leapfrog")].valid_steps)
            )
            cl = float(
                np.median(pe[(pe.n_traj == n) & (pe.arm == "closure")].valid_steps)
            )
            det[int(n)] = {
                "residual": r,
                "hnn_leapfrog": lf,
                "closure": cl,
                "leapfrog_over_residual": lf / max(r, 1.0),
            }
        ok = (
            det[ns[0]]["leapfrog_over_residual"] > det[ns[-1]]["leapfrog_over_residual"]
        )
    out["E9"] = {"verdict": _verdict(ok), "detail": det}
    for k, note in caveats(rows).items():
        out.setdefault(k, {})["caveats"] = note
    return out


def caveats(rows: pd.DataFrame) -> dict[str, list[str]]:
    """Findings next to each verdict that its criterion does not look at.
    Added after the criteria were fixed; they do not change a verdict."""
    out: dict[str, list[str]] = {}
    has = set(rows.system)
    n = []
    for s, k in (("pendulum", "H"), ("kepler", "E")):
        if s in has:
            n.append(
                f"{s}: {k} growth RK4-HNN {_med(rows, s, 'hnn', f'{k}_growth'):.3g}, "
                f"leapfrog {_med(rows, s, 'hnn_leapfrog', f'{k}_growth'):.3g}; "
                f"late error RK4-HNN {_med(rows, s, 'hnn', f'{k}_late'):.3g}, "
                f"leapfrog {_med(rows, s, 'hnn_leapfrog', f'{k}_late'):.3g}"
            )
    out["E2"] = n
    n = []
    for s in ("pendulum", "duffing", "lorenz", "burgers"):
        if s not in has:
            continue
        bb = BLACK_BOX if s in ODE_SYSTEMS else PDE_ARMS
        c = _med(rows, s, "closure", "valid_steps")
        best = max(bb, key=lambda a: _med(rows, s, a, "valid_steps"))
        n.append(
            f"{s}: in-distribution valid steps closure {c:.3g}, best black box "
            f"{best} {_med(rows, s, best, 'valid_steps'):.3g}"
        )
    out["E5"] = n
    ob = rows[rows.exp == "objective"]
    better: list[str] = []
    worse: list[str] = []
    for (s, a), _ in ob.groupby(["system", "arm"]):
        one = _med(rows, s, a, "valid_steps")
        unr = _med(rows, s, a, "valid_steps", exp="objective")
        (better if unr > one else worse).append(f"{s}/{a}")
    out["E6"] = [
        f"longer with unrolled loss: {', '.join(better) or 'none'}",
        f"not longer: {', '.join(worse) or 'none'}",
    ]
    n = []
    for s in PDE_SYSTEMS:
        if s not in has:
            continue
        for split in ("test", "ood"):
            c = _med(rows, s, "conv", "err_end", split)
            d = _med(rows, s, "dense", "err_end", split)
            cd = _med(rows, s, "conv", "diverged_frac", split)
            n.append(
                f"{s} ({split}): err at end conv {c:.3g}, dense {d:.3g}; "
                f"fraction of conv rollouts ending with error > 1: {cd:.3g}"
            )
    out["E7"] = n
    return out


def summary_table(
    rows: pd.DataFrame, kind: str = "ode", split: str = "test"
) -> pd.DataFrame:
    """Median over seeds per (system, arm), for the docs and the notebook."""
    cols = [
        "onestep",
        "err_train_h",
        "err_end",
        "valid_steps",
        "valid_over_train",
        "blowup_frac",
        "diverged_frac",
        "train_seconds",
        "n_params",
    ]
    extra = [
        c
        for c in rows.columns
        if c.endswith(("_late", "_growth"))
        or c in ("vpt_lyap", "mass_drift_end", "energy_ratio_end")
    ]
    m = rows[rows.exp.isin(["main", "ref"]) & (rows.split == split)]
    systems = ODE_SYSTEMS if kind == "ode" else PDE_SYSTEMS
    m = m[m.system.isin(systems)]
    use = [c for c in cols + extra if c in m.columns]
    g = m.groupby(["system", "arm"])[use].median().reset_index()
    if "n_params" in g:
        g["n_params"] = g["n_params"].astype(int)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    sys_order = {s: i for i, s in enumerate(systems)}
    g = g.sort_values(
        ["system", "arm"],
        key=lambda c: c.map(sys_order if c.name == "system" else order),
    )
    return g.dropna(axis=1, how="all").reset_index(drop=True)


def replace_opts(o: StudyOptions, **kw) -> StudyOptions:
    return replace(o, **kw)
