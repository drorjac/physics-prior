"""The Lorenz study: experiments, seeds, and the files they write.

Stages
------
tune      on the tuning seeds 3 / 7 / 19 only: the black box's size and
          features, the PINN's physics weight and Fourier features, the
          number of multiple-shooting segments. Chosen by median state error
          (constant error for shooting), then frozen.
main      the reference problem (40 observations, 5 % noise, three Lyapunov
          times) on the reporting seeds 11 / 23 / 42, every arm.
noise     noise from 0 to 20 %, 40 observations.
budget    10 to 160 observations, 5 % noise.
ladder    the PINN recipe built up one piece at a time, on the reporting seeds.
speed     milliseconds per training step for each way of computing du/dt.
chaos     separation of nearby trajectories, the Lyapunov exponent, the RK4
          step study, and the forecast horizon as a function of how wrong the
          constants are.

Arms
----
nn          black-box MLP on the observations
nn_regress  constants by regressing the black box's derivative on the law
fd_regress  constants by regressing finite differences of the raw data
pinn        the physics-informed network, constants trainable
shooting    single shooting from THETA_INIT
ms          multiple shooting from THETA_INIT
pinn_polish single shooting started from the PINN's answer

Every task is cached under .cache/lorenz keyed on its full description, so an
interrupted run resumes where it stopped. Runs go to a process pool with one
torch thread per worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import save_json, save_table
from physprior.logging import get_logger
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS

from . import TRACK
from . import classical as C
from . import metrics as M
from . import system as S
from .pinn import NN_DEFAULT, PINN_DEFAULT, PinnConfig, fit

log = get_logger(__name__)

EXAMPLE_SEED = 11
REF_NOISE = 0.05
REF_NOBS = 40
NOISES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
BUDGETS = (10, 20, 40, 80, 160)

# Tuning grids. The PINN grid includes w_phys = 10, where the fit collapses
# onto a law-satisfying trajectory that ignores the data; it stays in the grid
# so the collapse is on record.
NN_GRID: dict[str, dict] = {
    "w32_d3": {"width": 32, "depth": 3},
    "w64_d4": {"width": 64, "depth": 4},
    "w128_d3": {"width": 128, "depth": 3},
    "w64_d4_fourier8": {"width": 64, "depth": 4, "fourier": 8},
    "w64_d4_wd1e-4": {"width": 64, "depth": 4, "weight_decay": 1e-4},
}
PINN_GRID: dict[str, dict] = {
    f"w{w:g}_f{f}": {"w_phys": w, "fourier": f}
    for w in (0.1, 1.0, 10.0)
    for f in (0, 8)
}
MS_GRID = (6, 12, 24)

# The optimisation ladder. Each rung adds one thing to the one before it; the
# last two replace or add to the final recipe with the two loss-weighting
# schemes from the literature.
VANILLA = replace(
    PINN_DEFAULT,
    fourier=0,
    schedule="constant",
    warmup=0,
    ramp=0,
    resample=False,
    lbfgs=0,
    derivative="autograd",
)
LADDER: dict[str, dict] = {
    "vanilla": {},
    "+ warm-up and ramp": {"warmup": 1000, "ramp": 1000},
    "+ cosine lr, resampling": {
        "warmup": 1000,
        "ramp": 1000,
        "schedule": "cosine",
        "resample": True,
        "derivative": "forward",
    },
    "+ L-BFGS finish": {
        "warmup": 1000,
        "ramp": 1000,
        "schedule": "cosine",
        "resample": True,
        "derivative": "forward",
        "lbfgs": 1000,
    },
    "+ Fourier features": {
        "warmup": 1000,
        "ramp": 1000,
        "schedule": "cosine",
        "resample": True,
        "derivative": "forward",
        "lbfgs": 1000,
        "fourier": 8,
    },
    "vanilla + gradnorm": {"balance": "gradnorm"},
    "final + causal": {
        "warmup": 1000,
        "ramp": 1000,
        "schedule": "cosine",
        "resample": True,
        "derivative": "forward",
        "lbfgs": 1000,
        "fourier": 8,
        "causal_eps": 1.0,
    },
}
LADDER_FINAL = "+ Fourier features"


@dataclass(frozen=True)
class Options:
    tune_seeds: tuple[int, ...] = TUNE_SEEDS
    seeds: tuple[int, ...] = REPORT_SEEDS
    noises: tuple[float, ...] = NOISES
    budgets: tuple[int, ...] = BUDGETS
    steps: int = PINN_DEFAULT.steps
    lbfgs: int = PINN_DEFAULT.lbfgs
    warmup: int = PINN_DEFAULT.warmup
    ramp: int = PINN_DEFAULT.ramp
    ms_nfev: int = 300
    speed_steps: int = 150
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    quick: bool = False

    @classmethod
    def quick_options(cls) -> Options:
        return cls(
            tune_seeds=(3,),
            seeds=(11,),
            noises=(0.0, 0.05, 0.2),
            budgets=(20, 40),
            steps=600,
            lbfgs=50,
            warmup=150,
            ramp=150,
            ms_nfev=40,
            speed_steps=20,
            quick=True,
        )

    def scale(self, cfg: PinnConfig) -> PinnConfig:
        """Shorten a configuration in quick mode; identity otherwise."""
        if not self.quick:
            return cfg
        return replace(
            cfg,
            steps=self.steps,
            lbfgs=min(cfg.lbfgs, self.lbfgs),
            warmup=min(cfg.warmup, self.warmup),
            ramp=min(cfg.ramp, self.ramp),
        )


# ---------------------------------------------------------------------------
# tasks


@dataclass(frozen=True)
class Task:
    stage: str
    arm: str  # "nn" | "pinn" | "ms" | "shooting"
    label: str  # config name within the stage
    seed: int
    n_obs: int = REF_NOBS
    noise: float = REF_NOISE
    config: dict = field(default_factory=dict, hash=False)
    polish: bool = False
    keep: bool = False  # return the full example (histories, curves)

    def key(self) -> str:
        d = asdict(self)
        d.pop("stage")
        d.pop("keep")
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cache_dir() -> Path:
    p = get_settings().cache_dir / "lorenz"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _net_scores(obs: S.Observations, f) -> dict[str, Any]:
    pred = f.predict(obs.t_dense)
    du = f.derivative(obs.t_dense)
    out: dict[str, Any] = {
        "state": M.nrmse(pred, obs.u_dense),
        "deriv": M.nrmse(du, obs.du_dense),
        "seconds": f.seconds,
        "ms_per_step": f.ms_per_step,
    }
    u_end = f.predict(np.array([obs.t_end]))[0]
    if f.theta is not None:
        out |= {f"theta_{k}": v for k, v in f.theta_dict().items()}
        out |= M.theta_errors(f.theta)
        out |= M.forecast_scores(obs, M.forecast_law(obs, u_end, f.theta))
    else:
        t_f, _ = M.future(obs)
        out |= M.forecast_scores(obs, f.predict(t_f))
    return out


def _shooting_scores(obs: S.Observations, sf: C.ShootingFit) -> dict[str, Any]:
    pred = sf.predict(obs.t_dense)
    out: dict[str, Any] = {
        "state": M.nrmse(pred, obs.u_dense),
        "deriv": M.nrmse(S.lorenz_rhs(pred, sf.theta), obs.du_dense)
        if np.isfinite(pred).all()
        else float("inf"),
        "seconds": sf.seconds,
        "cost": sf.cost,
        "nfev": sf.nfev,
    }
    out |= {
        f"theta_{k}": float(v) for k, v in zip(S.THETA_NAMES, sf.theta, strict=True)
    }
    out |= M.theta_errors(sf.theta)
    if np.isfinite(pred).all():
        out |= M.forecast_scores(obs, M.forecast_law(obs, pred[-1], sf.theta))
    else:
        out["vpt"] = 0.0
    return out


def _example(obs, f, rows: dict) -> dict:
    """What the figures need from one fit: curves on the dense grid, the
    training history, the forecast error curve."""
    ex: dict[str, Any] = {
        "t_dense": obs.t_dense,
        "u_dense": obs.u_dense,
        "t_obs": obs.t_obs,
        "y_obs": obs.y_obs,
        "pred": f.predict(obs.t_dense),
        "deriv": f.derivative(obs.t_dense),
        "du_true": obs.du_dense,
        "history": f.history,
    }
    t_f, u_f = M.future(obs)
    if f.theta is not None:
        u_end = f.predict(np.array([obs.t_end]))[0]
        fc = M.forecast_law(obs, u_end, f.theta)
    else:
        fc = f.predict(t_f)
    ex["forecast_t"], ex["forecast_err"] = M.forecast_curve(obs, fc)
    ex["forecast_pred"] = fc
    ex["future_u"] = u_f
    return ex


def run_task(task: Task) -> dict:
    t0 = time.perf_counter()
    obs = S.observe(task.seed, task.n_obs, task.noise)
    base = {
        "stage": task.stage,
        "arm": task.arm,
        "label": task.label,
        "seed": task.seed,
        "n_obs": task.n_obs,
        "noise": task.noise,
    }
    rows: list[dict] = []
    example: dict[str, Any] = {}
    if task.arm in ("nn", "pinn"):
        cfg = PinnConfig(**task.config)
        f = fit(obs, cfg, task.seed)
        rows.append(base | _net_scores(obs, f))
        if task.arm == "nn":
            # the black box's route to constants: differentiate, regress
            th = C.regress_theta(f.predict(obs.t_dense), f.derivative(obs.t_dense))
            u_end = f.predict(np.array([obs.t_end]))[0]
            r = base | {"arm": "nn_regress"} | M.theta_errors(th)
            r |= {
                f"theta_{k}": float(v) for k, v in zip(S.THETA_NAMES, th, strict=True)
            }
            r |= M.forecast_scores(obs, M.forecast_law(obs, u_end, th))
            rows.append(r)
        if task.polish and f.theta is not None:
            u0 = f.predict(np.array([0.0]))[0]
            sf = C.polish(obs, f.theta, u0, "pinn_polish")
            rows.append(base | {"arm": "pinn_polish"} | _shooting_scores(obs, sf))
        if task.keep:
            example = _example(obs, f, rows[0])
    elif task.arm == "ms":
        sf = C.multiple_shooting(obs, **task.config)
        rows.append(base | _shooting_scores(obs, sf))
    elif task.arm == "shooting":
        sf = C.single_shooting(obs, **task.config)
        rows.append(base | _shooting_scores(obs, sf))
    elif task.arm == "fd":
        th = C.regress_theta(obs.y_obs, C.fd_derivative(obs.t_obs, obs.y_obs))
        r = base | {"arm": "fd_regress"} | M.theta_errors(th)
        r |= {f"theta_{k}": float(v) for k, v in zip(S.THETA_NAMES, th, strict=True)}
        rows.append(r)
    else:
        raise ValueError(f"unknown arm {task.arm!r}")
    for r in rows:
        r["wall"] = time.perf_counter() - t0
    return {"rows": rows, "example": example}


def _load_cached(task: Task) -> dict | None:
    p = _cache_dir() / f"{task.key()}.json"
    if not p.exists():
        return None
    out = json.loads(p.read_text())
    if task.keep and not out.get("example"):
        return None
    for r in out["rows"]:
        r["stage"] = task.stage
    return out


def _store(task: Task, out: dict) -> None:
    from physprior.io import _default

    p = _cache_dir() / f"{task.key()}.json"
    p.write_text(json.dumps(out, default=_default))


def execute(tasks: list[Task], workers: int) -> list[dict]:
    """Run tasks (cached ones are read back), longest first."""
    todo, done = [], []
    seen: set[str] = set()
    for t in tasks:
        c = _load_cached(t)
        if c is not None:
            done.append((t, c))
        elif t.key() not in seen:
            seen.add(t.key())
            todo.append(t)
        elif t.keep:
            todo = [t if u.key() == t.key() else u for u in todo]
    log.info("lorenz: %d tasks, %d cached, %d to run", len(tasks), len(done), len(todo))
    results = {t.key(): c for t, c in done}
    order = sorted(todo, key=lambda t: t.arm != "pinn")
    if order:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(order)), initializer=_init_worker
        ) as ex:
            futs = {ex.submit(run_task, t): t for t in order}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                out = fut.result()
                _store(t, out)
                results[t.key()] = out
                log.info(
                    "lorenz [%d/%d] %s %s %s seed=%d",
                    i,
                    len(order),
                    t.stage,
                    t.arm,
                    t.label,
                    t.seed,
                )
    # a duplicate that did not keep its example borrows the one that did
    outs = []
    for t in tasks:
        out = results[t.key()]
        rows = [dict(r, stage=t.stage) for r in out["rows"]]
        outs.append({"rows": rows, "example": out["example"] if t.keep else {}})
    return outs


def _rows(outs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([r for o in outs for r in o["rows"]])


# ---------------------------------------------------------------------------
# stages


def _net_config(o: Options, base: PinnConfig, **kw) -> dict:
    return asdict(o.scale(replace(base, **kw)))


def tune(o: Options) -> tuple[pd.DataFrame, dict]:
    tasks = []
    for s in o.tune_seeds:
        for name, kw in NN_GRID.items():
            tasks.append(
                Task("tune", "nn", name, s, config=_net_config(o, NN_DEFAULT, **kw))
            )
        for name, kw in PINN_GRID.items():
            tasks.append(
                Task("tune", "pinn", name, s, config=_net_config(o, PINN_DEFAULT, **kw))
            )
        for n in MS_GRID:
            tasks.append(
                Task(
                    "tune",
                    "ms",
                    f"seg{n}",
                    s,
                    config={"n_seg": n, "max_nfev": o.ms_nfev},
                )
            )
    df = _rows(execute(tasks, o.workers))
    choice = {}
    for arm, col in (("nn", "state"), ("pinn", "state"), ("ms", "err_theta")):
        med = df[df.arm == arm].groupby("label")[col].median()
        choice[arm] = str(med.idxmin())
    return df, choice


def chosen_configs(o: Options, choice: dict) -> dict[str, dict]:
    return {
        "nn": _net_config(o, NN_DEFAULT, **NN_GRID[choice["nn"]]),
        "pinn": _net_config(o, PINN_DEFAULT, **PINN_GRID[choice["pinn"]]),
        "ms": {"n_seg": int(choice["ms"].removeprefix("seg")), "max_nfev": o.ms_nfev},
    }


def _arm_tasks(stage, cfgs, seed, n_obs, noise, keep=False, polish=False, extra=()):
    ts = [
        Task(stage, "nn", "chosen", seed, n_obs, noise, cfgs["nn"], keep=keep),
        Task(
            stage,
            "pinn",
            "chosen",
            seed,
            n_obs,
            noise,
            cfgs["pinn"],
            polish=polish,
            keep=keep,
        ),
        Task(stage, "ms", "chosen", seed, n_obs, noise, cfgs["ms"]),
    ]
    ts += [Task(stage, a, "default", seed, n_obs, noise, c) for a, c in extra]
    return ts


def main_tasks(o: Options, cfgs: dict) -> list[Task]:
    extra = [("shooting", {"max_nfev": o.ms_nfev}), ("fd", {})]
    return [
        t
        for s in o.seeds
        for t in _arm_tasks(
            "main",
            cfgs,
            s,
            REF_NOBS,
            REF_NOISE,
            keep=s == EXAMPLE_SEED,
            polish=True,
            extra=extra,
        )
    ]


def noise_tasks(o: Options, cfgs: dict) -> list[Task]:
    return [
        t
        for s in o.seeds
        for eta in o.noises
        for t in _arm_tasks(
            "noise", cfgs, s, REF_NOBS, eta, polish=True, extra=[("fd", {})]
        )
    ]


def budget_tasks(o: Options, cfgs: dict) -> list[Task]:
    return [
        t
        for s in o.seeds
        for n in o.budgets
        for t in _arm_tasks(
            "budget", cfgs, s, n, REF_NOISE, polish=True, extra=[("fd", {})]
        )
    ]


def ladder_tasks(o: Options, cfgs: dict) -> list[Task]:
    w = cfgs["pinn"]["w_phys"]
    return [
        Task(
            "ladder",
            "pinn",
            name,
            s,
            config=asdict(o.scale(replace(VANILLA, w_phys=w, **kw))),
            # the vanilla rung keeps every seed: it fails on some, and the
            # figure of its loss has to show one where it does
            keep=s == EXAMPLE_SEED or name == "vanilla",
        )
        for s in o.seeds
        for name, kw in LADDER.items()
    ]


# ---------------------------------------------------------------------------
# speed and chaos, run in the parent process


def speed(o: Options, cfg: dict) -> pd.DataFrame:
    """ms per Adam step at the study's size, for each derivative method,
    one thread, float64. Median of three repeats."""
    import torch

    torch.set_num_threads(1)
    obs = S.observe(EXAMPLE_SEED)
    rows = []
    variants = [
        ("autograd", False),
        ("jvp", False),
        ("forward", False),
        ("forward", True),
    ]
    for mode, comp in variants:
        base = replace(
            PinnConfig(**cfg),
            steps=o.speed_steps,
            warmup=0,
            ramp=0,
            lbfgs=0,
            derivative=mode,
            compile=comp,
            record_every=10**9,
        )
        times = []
        try:
            fit(obs, replace(base, steps=5), EXAMPLE_SEED)  # warm-up / compile
            for rep in range(3):
                times.append(fit(obs, base, EXAMPLE_SEED + rep).ms_per_step)
            err = ""
        except Exception as e:  # torch.compile needs a working C++ toolchain
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        rows.append(
            {
                "derivative": mode,
                "compiled": comp,
                "ms_per_step": float(np.median(times)) if times else float("nan"),
                "error": err,
                "torch": torch.__version__,
            }
        )
    # agreement of the three derivative methods on one network
    from .pinn import LorenzNet, derivative_fn

    torch.manual_seed(0)
    net = LorenzNet(64, 4, 8)
    s = torch.linspace(-1, 1, 257, dtype=torch.float64)[:, None]
    ref = derivative_fn(net, "autograd")(s)[1].detach()
    for r in rows:
        d = derivative_fn(net, str(r["derivative"]))(s)[1].detach()
        r["max_abs_diff_vs_autograd"] = float((d - ref).abs().max())
    return pd.DataFrame(rows)


def chaos(o: Options) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separation growth, the RK4 step study, and the forecast horizon as a
    function of the relative error in all three constants."""
    u0 = S.attractor_state(EXAMPLE_SEED)
    t, a, b = S.separation(u0, delta=1e-8, t_end=40.0, n_out=4001)
    dist = np.linalg.norm(a - b, axis=1)
    # One pair's log-distance wanders with the local stretching rate; the
    # mean over pairs started across the attractor has the slope lambda_1.
    n_pairs = 4 if o.quick else 24
    logs = [np.log(dist)]
    for k in range(1, n_pairs):
        _, a_k, b_k = S.separation(
            S.attractor_state(1000 + k), delta=1e-8, t_end=40.0, n_out=4001
        )
        logs.append(np.log(np.linalg.norm(a_k - b_k, axis=1)))
    mean_log = np.mean(logs, axis=0)
    lam = S.growth_rate(t, np.exp(mean_log), 1e-7, 1e-1)
    summary = {
        "delta0": 1e-8,
        "n_pairs": n_pairs,
        "lambda_fit": lam,
        "lambda_single_pair": S.growth_rate(t, dist, 1e-7, 1e-1),
        "lambda_reference": S.LAMBDA1,
        "t_saturate": float(t[int(np.argmax(dist > 1.0))]),
    }
    sep = pd.DataFrame({"t": t, "dist": dist, "mean_log_dist": mean_log})
    conv = pd.DataFrame(S.rk4_convergence())
    # horizon vs constant error, from the exact state at the end of the window
    rows = []
    eps_grid = np.geomspace(1e-5, 3e-1, 14 if not o.quick else 5)
    for s in o.seeds:
        obs = S.observe(s)
        u_end = obs.u_dense[-1]
        for eps in eps_grid:
            for sign in (1.0, -1.0):
                th = S.THETA_TRUE * (1.0 + sign * eps)
                fc = M.forecast_law(obs, u_end, th)
                rows.append(
                    {"seed": s, "eps": eps, "sign": sign} | M.forecast_scores(obs, fc)
                )
    return summary, sep, pd.DataFrame(rows), conv


# ---------------------------------------------------------------------------


def run(quick: bool = False, opts: Options | None = None) -> dict:
    o = opts or (Options.quick_options() if quick else Options())
    t0 = time.perf_counter()
    tuning, choice = tune(o)
    cfgs = chosen_configs(o, choice)
    save_table(tuning, TRACK, "tuning")
    log.info("lorenz: chosen %s", choice)

    tasks = {
        "main": main_tasks(o, cfgs),
        "noise": noise_tasks(o, cfgs),
        "budget": budget_tasks(o, cfgs),
        "ladder": ladder_tasks(o, cfgs),
    }
    flat = [t for ts in tasks.values() for t in ts]
    outs = execute(flat, o.workers)
    frames = _rows(outs)
    for stage in tasks:
        save_table(frames[frames.stage == stage].reset_index(drop=True), TRACK, stage)

    examples = {}
    for t, out in zip(flat, outs, strict=True):
        if t.keep and out.get("example"):
            examples[f"{t.stage}/{t.arm}/{t.label}/{t.seed}"] = out["example"]
    save_examples(examples)

    sp = speed(o, cfgs["pinn"])
    save_table(sp, TRACK, "speed")
    summary, sep, horizon, conv = chaos(o)
    save_table(sep.iloc[::10].reset_index(drop=True), TRACK, "separation")
    save_table(horizon, TRACK, "horizon")
    save_table(conv, TRACK, "rk4_convergence")

    meta = {
        "choice": choice,
        "configs": cfgs,
        "options": {k: v for k, v in asdict(o).items() if k != "workers"},
        "chaos": summary,
        "reference": {"n_obs": REF_NOBS, "noise": REF_NOISE, "t_end": 3.0},
        "theta_true": dict(zip(S.THETA_NAMES, S.THETA_TRUE.tolist(), strict=True)),
        "theta_init": dict(zip(S.THETA_NAMES, S.THETA_INIT.tolist(), strict=True)),
        "seconds": time.perf_counter() - t0,
    }
    save_json(meta, TRACK, "meta")
    log.info("lorenz: done in %.0f s", meta["seconds"])
    return meta


def save_examples(examples: dict) -> None:
    """Arrays for the figures, as one compressed npz next to the CSVs."""
    flat: dict[str, Any] = {}
    for key, ex in examples.items():
        for k, v in ex.items():
            if k == "history":
                for hk, hv in v.items():
                    arr = np.asarray(hv)
                    flat[f"{key}|history|{hk}"] = arr
            else:
                flat[f"{key}|{k}"] = np.asarray(v, dtype=float)
    path = get_settings().results(TRACK) / "examples.npz"
    np.savez_compressed(path, **flat)


def load_examples() -> dict[str, dict]:
    path = get_settings().results_dir / TRACK / "examples.npz"
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict] = {}
    for name in data.files:
        parts = name.split("|")
        ex = out.setdefault(parts[0], {"history": {}})
        if parts[1] == "history":
            ex["history"][parts[2]] = data[name]
        else:
            ex[parts[1]] = data[name]
    return out


def load(stage: str) -> pd.DataFrame:
    from physprior.io import load_table

    return load_table(TRACK, stage)
