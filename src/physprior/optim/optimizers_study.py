"""Optimizers x learning rates x model families x problems.

The question: does the way a model is optimized depend on whether it carries
physics? Three expectations, each tested here rather than assumed:

1. A closed-form law with 1-3 constants has a low-dimensional loss whose
   Hessian at the minimum is small and usually well conditioned. A
   second-order method (Gauss-Newton / Levenberg-Marquardt, or L-BFGS)
   should finish in tens of evaluations where first-order methods need
   thousands.
2. A black-box network is overparameterised: at a minimum most Hessian
   eigenvalues are ~0 (flat directions), with a few large ones (sharp
   directions). Adam-type methods cope; plain SGD is capped by the sharpest
   direction (lr < 2 / lambda_max) and crawls along the rest.
3. A residual PINN's loss is ill-conditioned: the residual involves
   derivatives of the network, which amplify high frequencies, so the physics
   term's curvature and gradients dwarf the data term's (Krishnapriyan et al.
   2021; Wang, Teng & Perdikaris 2021). Measured here as the Hessian spectrum
   of each loss term and the ratio of gradient norms during training.

Protocol. The learning-rate grid is run on the TUNING seeds; the best rate
per (task, model, optimizer) is chosen there by median final data loss, and
only that rate is run on the REPORTING seeds for the reported table. The
lr-grid table is itself a result (sensitivity and failure rate), and it says
which seeds it came from.

Budget is counted in loss-and-gradient evaluations, so L-BFGS's line-search
evaluations are charged like any other step.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from physprior.benchmark.metrics import nrmse
from physprior.io import _default as _io_default
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS
from physprior.optim.problems import (
    MODELS,
    TASKS,
    Model,
    full_hessian,
    get_task,
    spectrum_summary,
    top_eigenvalues,
)

AREA = "optim"
OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam", "lbfgs")
LR_GRID = (1e-3, 1e-2, 1e-1, 1.0)
LBFGS_LR_GRID = (0.1, 1.0)
# Evaluation budgets. The residual PINNs are the expensive ones: each
# evaluation differentiates the network twice with respect to its input.
BUDGET = {"hydrogen": 2000, "oscillator": 1500, "heat": 1000}
QUICK_BUDGET = 150
RECORD_EVERY = 20
# A loss that grows past this multiple of its starting value has diverged.
DIVERGE_FACTOR = 1e6
FULL_HESSIAN_MAX = 2500


@dataclass(frozen=True)
class RunSpec:
    task: str
    model: str
    optimizer: str
    lr: float
    seed: int
    budget: int
    record: bool = False
    curvature: bool = False


def _make_opt(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )
    raise ValueError(f"unknown optimizer {name!r}")


def grad_norms(m: Model) -> tuple[float, float]:
    """||grad L_data|| and ||grad w L_phys|| over the NETWORK weights -- the
    shared parameters the two terms compete for."""
    ps = m.net_params()
    if not ps or m.kind != "pinn":
        return float("nan"), float("nan")
    gd = torch.autograd.grad(m.data_loss(), ps, allow_unused=True)
    gp = torch.autograd.grad(m.w_phys * m.phys_loss(), ps, allow_unused=True)

    def norm(gs):
        return float(
            torch.sqrt(sum(torch.sum(g**2) for g in gs if g is not None))  # type: ignore[arg-type]
        )

    return norm(gd), norm(gp)


def run_one(spec: RunSpec) -> tuple[dict, list[dict], dict | None]:
    """Train one model with one optimizer. Returns (row, curve, curvature)."""
    torch.set_num_threads(1)
    task = get_task(spec.task)
    m = Model(task, spec.model, spec.seed)
    params = m.parameters()
    opt = _make_opt(spec.optimizer, params, spec.lr)
    t0 = time.time()

    state: dict[str, Any] = {
        "evals": 0,
        "first_tol": None,
        "diverged": False,
        "l0": None,
    }
    curve: list[dict] = []

    def record(total: float, data: float, phys: float) -> None:
        e = state["evals"]
        if not spec.record or e % RECORD_EVERY:
            return
        row = {"evals": e, "loss": total, "data_loss": data, "phys_loss": phys}
        row.update({f"theta_{k}": v for k, v in m.theta().items()})
        if m.kind == "pinn":
            gd, gp = grad_norms(m)
            row.update(grad_data=gd, grad_phys=gp)
        curve.append(row)

    def evaluate() -> torch.Tensor:
        data = m.data_loss()
        phys = m.phys_loss() if m.kind == "pinn" else torch.zeros((), dtype=data.dtype)
        loss = data + m.w_phys * phys if m.kind == "pinn" else data
        d, lv = float(data.detach()), float(loss.detach())
        if state["l0"] is None:
            state["l0"] = lv
        if not np.isfinite(lv) or lv > DIVERGE_FACTOR * max(state["l0"], 1e-12):
            state["diverged"] = True
        if state["first_tol"] is None and d <= task.tol:
            state["first_tol"] = state["evals"]
        record(lv, d, float(phys.detach()))
        state["evals"] += 1
        return loss

    if spec.optimizer == "lbfgs":

        def closure():
            opt.zero_grad()
            loss = evaluate()
            if state["diverged"]:
                return loss.detach()
            loss.backward()
            return loss

        while state["evals"] < spec.budget and not state["diverged"]:
            before = state["evals"]
            try:
                opt.step(closure)
            except RuntimeError:
                state["diverged"] = True
            if state["evals"] == before:  # converged: L-BFGS stops calling
                break
    else:
        while state["evals"] < spec.budget and not state["diverged"]:
            opt.zero_grad()
            loss = evaluate()
            if state["diverged"]:
                break
            loss.backward()
            opt.step()

    seconds = time.time() - t0
    with torch.no_grad():
        final_data = float(m.data_loss())
    final_phys = float(m.phys_loss().detach()) if m.kind == "pinn" else 0.0
    theta = m.theta()
    finite = np.isfinite(final_data) and not state["diverged"]
    row: dict[str, Any] = {
        "task": spec.task,
        "model": spec.model,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "seed": spec.seed,
        "budget": spec.budget,
        "evals_used": state["evals"],
        "n_params": m.n_params(),
        "tol": task.tol,
        "final_data_loss": final_data if finite else np.inf,
        "final_phys_loss": final_phys,
        "evals_to_tol": state["first_tol"],
        "reached_tol": state["first_tol"] is not None
        and finite
        and final_data <= 10 * task.tol,
        "diverged": bool(state["diverged"] or not np.isfinite(final_data)),
        "seconds": seconds,
    }
    for label, x, y in (("in", task.x_in, task.y_in), ("out", task.x_out, task.y_out)):
        pred = m.predict(x) if finite else np.full(len(y), np.nan)
        row[f"nrmse_{label}"] = (
            nrmse(y, pred, scale=task.scale) if np.all(np.isfinite(pred)) else np.inf
        )
    for k, v in theta.items():
        row[f"theta_{k}"] = v
        if k in task.truth:
            row[f"err_{k}_pct"] = (v - task.truth[k]) / task.truth[k] * 100.0
    row["param_err_pct"] = (
        abs(row.get(f"err_{task.key_param}_pct", np.nan)) if theta else np.nan
    )
    curv = curvature(m) if spec.curvature and finite else None
    for c in curve:
        c.update(
            task=spec.task, model=spec.model, optimizer=spec.optimizer, seed=spec.seed
        )
    return row, curve, curv


def curvature(m: Model, k_top: int = 12) -> dict:
    """Hessian spectrum of the training loss at the current point, and of
    each loss term separately for a PINN."""
    params = m.parameters()
    n = m.n_params()
    out: dict = {"task": m.task.name, "model": m.kind, "n_params": n}
    # A residual PINN's HVP differentiates the network three times; forming
    # its full Hessian costs thousands of those, so it gets Lanczos instead.
    residual = m.kind == "pinn" and m.task.pinn_shape == "residual"
    if n <= FULL_HESSIAN_MAX and not residual:
        eig = np.linalg.eigvalsh(full_hessian(m.loss, params))[::-1]
        out["method"] = "full"
        out.update(spectrum_summary(eig))
    else:
        eig = top_eigenvalues(m.loss, params, k=k_top)
        out["method"] = "lanczos"
        out.update(spectrum_summary(eig))
        # Only the top of the spectrum is known: no statement about the rest.
        for k in ("lambda_min", "frac_constrained", "cond_constrained", "n_negative"):
            out[k] = float("nan")
        out["n_eig"] = n
    out["eigenvalues"] = eig.tolist()
    if m.kind == "pinn":
        net = m.net_params()
        out["lambda_max_data_net"] = float(top_eigenvalues(m.data_loss, net, k=1)[0])
        out["lambda_max_phys_net"] = float(
            top_eigenvalues(lambda: m.w_phys * m.phys_loss(), net, k=1)[0]
        )
        if m.phys_params():
            ph = m.phys_params()
            Hp = full_hessian(m.loss, ph)
            ev = np.linalg.eigvalsh(Hp)
            out["phys_block_cond"] = float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf
    if m.kind == "physics":
        ev = np.sort(eig)
        out["phys_block_cond"] = float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf
    return out


# ---------------------------------------------------------------------------
# Levenberg-Marquardt, the reference second-order method for `physics`
# ---------------------------------------------------------------------------


def run_lm(task_name: str, seed: int) -> dict:
    """scipy's MINPACK LM on the same residuals, same starting point, the
    same parameterisation (log of the constant relative to its initial
    guess). Evaluations are counted as function calls plus Jacobian calls,
    each Jacobian costing n_params function calls by finite differences."""
    from scipy.optimize import least_squares

    task = get_task(task_name)
    m = Model(task, "physics", seed)
    raw = list(m.phys_params())
    x0 = np.array([float(r.detach()) for r in raw])
    count = {"n": 0, "first_tol": None}

    def resid(z):
        with torch.no_grad():
            for r, v in zip(raw, z, strict=True):
                r.fill_(float(v))
            res = ((m.forward(m.x) - m.y) / task.sd_y).numpy() / np.sqrt(
                len(task.y_train)
            )
        count["n"] += 1
        d = float(np.sum(res**2))
        if count["first_tol"] is None and np.isfinite(d) and d <= task.tol:
            count["first_tol"] = count["n"]
        return np.where(np.isfinite(res), res, 1e6)

    t0 = time.time()
    sol = least_squares(resid, x0, method="lm", xtol=1e-15, ftol=1e-15, max_nfev=2000)
    resid(sol.x)
    final = float(np.sum(sol.fun**2))
    theta = m.theta()
    row: dict[str, Any] = {
        "task": task_name,
        "model": "physics",
        "optimizer": "lm",
        "lr": np.nan,
        "seed": seed,
        "budget": 2000,
        "evals_used": count["n"],
        "n_params": len(x0),
        "tol": task.tol,
        "final_data_loss": final,
        "final_phys_loss": 0.0,
        "evals_to_tol": count["first_tol"],
        "reached_tol": count["first_tol"] is not None and final <= 10 * task.tol,
        "diverged": not np.isfinite(final),
        "seconds": time.time() - t0,
    }
    for label, x, y in (("in", task.x_in, task.y_in), ("out", task.x_out, task.y_out)):
        row[f"nrmse_{label}"] = nrmse(y, m.predict(x), scale=task.scale)
    for k, v in theta.items():
        row[f"theta_{k}"] = v
        row[f"err_{k}_pct"] = (v - task.truth[k]) / task.truth[k] * 100.0
    row["param_err_pct"] = abs(row[f"err_{task.key_param}_pct"])
    return row


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------


def _default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    return _io_default(o)


def _cache_path(name: str):
    from physprior.config import get_settings

    d = get_settings().cache_dir / "optim"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.jsonl"


def _pool_map(specs: list[RunSpec], workers: int, cache: str | None = None):
    """Run the specs, in parallel if asked. With `cache`, each finished run
    is appended to a JSON-lines file under the cache directory and skipped
    on a rerun, so an interrupted study resumes where it stopped."""
    done: dict[str, tuple] = {}
    path = _cache_path(cache) if cache else None
    if path is not None and path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = tuple(rec["result"])
    todo = [s for s in specs if repr(s) not in done]
    if todo:
        print(f"  {len(todo)} runs to do, {len(specs) - len(todo)} cached", flush=True)

    def store(spec, result):
        done[repr(spec)] = result
        if path is not None:
            with path.open("a") as fh:
                fh.write(
                    json.dumps({"key": repr(spec), "result": result}, default=_default)
                    + "\n"
                )

    if workers <= 1:
        for sp in todo:
            store(sp, run_one(sp))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(run_one, sp): sp for sp in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                store(futs[fut], fut.result())
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)} runs", flush=True)
    return [done[repr(sp)] for sp in specs]


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def lr_grid(opt: str) -> tuple[float, ...]:
    return LBFGS_LR_GRID if opt == "lbfgs" else LR_GRID


def select_lr(grid: pd.DataFrame) -> pd.DataFrame:
    """Best rate per (task, model, optimizer) on the tuning seeds: lowest
    median final data loss, a diverged run counting as infinite."""
    g = grid.copy()
    g["score"] = g["final_data_loss"].where(~g["diverged"], np.inf)
    med = (
        g.groupby(["task", "model", "optimizer", "lr"])["score"].median().reset_index()
    )
    idx = med.groupby(["task", "model", "optimizer"])["score"].idxmin()
    return med.loc[idx].rename(columns={"score": "tune_median_data_loss"})


def run(
    quick: bool = False,
    tasks=TASKS,
    models=MODELS,
    optimizers=OPTIMIZERS,
    workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    from physprior.io import save_json, save_table

    workers = workers if workers is not None else (1 if quick else 3)
    tune_seeds = TUNE_SEEDS[:1] if quick else TUNE_SEEDS
    report_seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS

    def budget(t):
        return QUICK_BUDGET if quick else BUDGET[t]

    # 1. learning-rate grid on the tuning seeds
    specs = [
        RunSpec(t, m, o, lr, s, budget(t))
        for t in tasks
        for m in models
        for o in optimizers
        for lr in (lr_grid(o) if not quick else lr_grid(o)[-2:])
        for s in tune_seeds
    ]
    t0 = time.time()
    grid = pd.DataFrame(
        [r[0] for r in _pool_map(specs, workers, None if quick else "lr_grid")]
    )
    grid["phase"] = "tune"
    save_table(grid, AREA, "lr_grid")
    print(f"  lr grid: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)

    # 2. the chosen rate on the reporting seeds, with curves; curvature at
    #    the Adam and L-BFGS solutions
    chosen = select_lr(grid)
    save_table(chosen, AREA, "lr_chosen")
    specs = [
        RunSpec(
            r.task,
            r.model,
            r.optimizer,
            float(r.lr),
            s,
            budget(r.task),
            record=True,
            curvature=r.optimizer in ("adam", "lbfgs") and not quick,
        )
        for r in chosen.itertuples()
        for s in report_seeds
    ]
    t0 = time.time()
    res = _pool_map(specs, workers, None if quick else "report")
    report = pd.DataFrame([r[0] for r in res])
    lm_rows = [run_lm(t, s) for t in tasks for s in report_seeds if "physics" in models]
    report = pd.concat([report, pd.DataFrame(lm_rows)], ignore_index=True)
    report["phase"] = "report"
    curves = pd.DataFrame([c for r in res for c in r[1]])
    curv_rows = []
    for spec, r in zip(specs, res, strict=True):
        if r[2] is not None:
            curv_rows.append({"optimizer": spec.optimizer, "seed": spec.seed, **r[2]})
    print(f"  report: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)
    save_table(report, AREA, "optimizers")
    save_table(curves, AREA, "curves")
    spectra = pd.DataFrame(
        [{k: v for k, v in c.items() if k != "eigenvalues"} for c in curv_rows]
    )
    save_table(spectra, AREA, "curvature")
    save_json(
        [
            {k: c[k] for k in ("task", "model", "optimizer", "seed", "eigenvalues")}
            for c in curv_rows
        ],
        AREA,
        "hessian_eigenvalues",
    )
    return {"grid": grid, "report": report, "curves": curves, "curvature": spectra}


def summarise(report: pd.DataFrame) -> pd.DataFrame:
    """Median over reporting seeds per (task, model, optimizer)."""
    g = report.groupby(["task", "model", "optimizer"])
    out = g.agg(
        lr=("lr", "first"),
        n_params=("n_params", "first"),
        final_data_loss=("final_data_loss", "median"),
        evals_to_tol=("evals_to_tol", "median"),
        reached=("reached_tol", "mean"),
        diverged=("diverged", "mean"),
        nrmse_in=("nrmse_in", "median"),
        nrmse_out=("nrmse_out", "median"),
        param_err_pct=("param_err_pct", "median"),
        seconds=("seconds", "median"),
    ).reset_index()
    out["failure_rate"] = 1.0 - out["reached"]
    return out
