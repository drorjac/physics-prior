"""Choose each track's physics weight for the `pinn` arm, on the tuning seeds.

    physprior tune --what pinn_weight

The `pinn` arm is law + correction, trained on MSE/sd_y^2 + w_phys*mean(NN^2).
`w_phys` sets how much the correction may do. It used to be annealed during
training (loss balancing); that drove it up without bound and switched the
correction off (see `methods/pinn.py`, FROZEN_PINN). It is now a fixed
hyperparameter per track, chosen here and written into the track's
`problem()` as `pinn_w_phys`.

The rule, fixed before any weight was scored:

  * candidates: W_GRID, all positive (w_phys = 0 is a network wearing a law,
    and stays in the reported sweep as a measurement, not as a candidate);
  * data: only the track's extrapolation TRAINING set. Its top VAL_FRAC in
    the first input column is held out as validation, so the choice rewards
    a correction that carries past the range it was fitted on, which is the
    job the arm is scored on. The extrapolation split is the same on every
    seed, so scoring on its test points would select on reported data;
  * seeds: TUNE_SEEDS (3 / 7 / 19) only;
  * score: median over those seeds of the validation nRMSE, in units of the
    training targets' spread;
  * choice: the lowest median; any weight within TIE of it counts as tied,
    and ties go to the larger weight, the more constrained model;
  * edges: if the choice is an end of the grid, the grid is extended one
    decade past it and the choice repeated, up to MAX_EXTEND decades, so the
    chosen weight is an interior optimum or is reported as pinned. This rule
    was added after helium's first pass chose the smallest weight on the
    grid; it uses the same tuning seeds and the same validation block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.benchmark.metrics import nrmse
from physprior.benchmark.protocol import Problem, fit_arm, split_extrapolate
from physprior.methods.base import TUNE_SEEDS
from physprior.units import require

W_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
VAL_FRAC = 0.25
TIE = 0.02
MAX_EXTEND = 3


def inner_split(prob: Problem, train_idx: np.ndarray, val_frac: float = VAL_FRAC):
    """The top `val_frac` of the training range as validation, the rest to fit."""
    order = train_idx[np.argsort(prob.x[train_idx, 0], kind="stable")]
    k = max(1, round(val_frac * len(order)))
    fit, val = np.sort(order[:-k]), np.sort(order[-k:])
    require(len(fit) >= 2, f"{prob.track}: {len(fit)} points left to fit")
    return fit, val


def _score_grid(prob, fit_idx, val_idx, scale, grid, seeds) -> pd.DataFrame:
    rows = []
    for w in grid:
        for seed in seeds:
            f = fit_arm("pinn", prob, fit_idx, seed, w_phys=w)
            pred = np.asarray(f.predict(prob.x[val_idx]), float).ravel()
            rows.append(
                {
                    "track": prob.track,
                    "w_phys": w,
                    "seed": seed,
                    "n_fit": len(fit_idx),
                    "n_val": len(val_idx),
                    "val_nrmse": nrmse(prob.y[val_idx], pred, scale=scale),
                    "correction_rms_frac": f.extra.get("correction_rms_frac"),
                }
            )
    return pd.DataFrame(rows)


def _choose(df: pd.DataFrame) -> float:
    med = df.groupby("w_phys")["val_nrmse"].median()
    tied = med[med <= float(med.min()) * (1.0 + TIE)]
    return float(tied.index.max())


def select_w_phys(
    prob: Problem,
    train_idx: np.ndarray,
    grid=W_GRID,
    seeds=TUNE_SEEDS,
) -> tuple[float, pd.DataFrame]:
    fit_idx, val_idx = inner_split(prob, train_idx)
    scale = float(np.std(prob.y[train_idx]))
    df = _score_grid(prob, fit_idx, val_idx, scale, grid, seeds)
    chosen = _choose(df)
    for _ in range(MAX_EXTEND):
        lo, hi = float(df.w_phys.min()), float(df.w_phys.max())
        if chosen == lo:
            extra = lo / 10.0
        elif chosen == hi:
            extra = hi * 10.0
        else:
            break
        df = pd.concat(
            [df, _score_grid(prob, fit_idx, val_idx, scale, [extra], seeds)],
            ignore_index=True,
        )
        chosen = _choose(df)
    med = df.groupby("w_phys")["val_nrmse"].median()
    df["median_val_nrmse"] = df["w_phys"].map(med)
    df["chosen"] = df["w_phys"] == chosen
    df["pinned_at_edge"] = chosen in (float(df.w_phys.min()), float(df.w_phys.max()))
    return chosen, df


def train_split(track: str, prob: Problem) -> np.ndarray:
    """The extrapolation training set each track's `run()` uses."""
    from physprior.problems.quantum import cmb, helium

    x0 = prob.x[:, 0]
    frac = {
        "gravity/kepler": 0.5,
        "gravity/pulsar_spindown": 0.5,
        "quantum/hydrogen": 0.25,
        "quantum/cmb": float(np.mean(x0 < cmb.TURNOVER_HZ)),
        "quantum/helium": float(np.mean(x0 <= helium.N_TRAIN_MAX)),
    }[track]
    itr, _ = split_extrapolate(x0, frac)
    return itr


def problems() -> dict:
    from physprior.problems.gravity import kepler, pulsar_spindown
    from physprior.problems.quantum import cmb, helium, hydrogen

    return {
        "gravity/kepler": kepler.problem,
        "gravity/pulsar_spindown": pulsar_spindown.problem,
        "quantum/hydrogen": hydrogen.problem,
        "quantum/cmb": cmb.problem,
        "quantum/helium": helium.problem,
    }


def run(tracks=None) -> pd.DataFrame:
    """Select the weight for every track; returns the full scoring table."""
    frames = []
    for track, make in problems().items():
        if tracks and track not in tracks:
            continue
        prob, _ = make()
        chosen, df = select_w_phys(prob, train_split(track, prob))
        frames.append(df)
        print(f"  {track:20s} w_phys = {chosen:g}", flush=True)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Re-running the `pinn` rows after the arm's configuration changes
# --------------------------------------------------------------------------


def _replace_arm_rows(old: pd.DataFrame, new: pd.DataFrame, arm: str) -> pd.DataFrame:
    """`old` with its `arm` rows swapped for `new`, in the same positions.

    Every other arm's rows are kept byte for byte: their fits do not depend
    on the pinn's configuration, the noise draws depend only on the seed and
    the level, and the splits only on the seed.
    """
    drop = old.arm == arm
    if "track" in new and "track" in old:
        drop &= old.track.isin(set(new.track))
    keep = old[~drop]
    out = pd.concat([keep, new], ignore_index=True)
    key = [
        c
        for c in (
            "track",
            "block",
            "n_train",
            "noise_frac",
            "noise_db",
            "w_phys",
            "seed",
        )
        if c in out
    ]
    order = {a: i for i, a in enumerate(dict.fromkeys(old.arm))}
    out["_arm_order"] = out.arm.map(order)
    out = out.sort_values([*key, "_arm_order"], kind="stable").drop(
        columns="_arm_order"
    )
    return out[old.columns.union(new.columns, sort=False)].reset_index(drop=True)


def rerun_pinn(track: str, prob: Problem) -> dict:
    """Recompute the pinn rows of every committed sweep of `track`, with the
    settings each sweep was run with, read back from its own table."""
    from physprior.benchmark.protocol import (
        study_extrapolation,
        sweep_budget,
        sweep_noise,
        sweep_physics_weight,
    )
    from physprior.io import load_table, save_table

    arms = ("pinn",)
    out = {}

    b = load_table(track, "sweep_budget")
    new = sweep_budget(prob, sorted(b.n_train.unique()), arms=arms, progress=False)
    save_table(_replace_arm_rows(b, new, "pinn"), track, "sweep_budget")
    out["sweep_budget"] = len(new)

    nz = load_table(track, "sweep_noise")
    new = sweep_noise(
        prob,
        sorted(nz.noise_frac.unique()),
        arms=arms,
        n_train=int(nz.n_train.iloc[0]),
        progress=False,
    )
    save_table(_replace_arm_rows(nz, new, "pinn"), track, "sweep_noise")
    out["sweep_noise"] = len(new)

    ex = load_table(track, "extrapolation")
    new = study_extrapolation(prob, train_frac=float(ex.train_frac.iloc[0]), arms=arms)
    save_table(_replace_arm_rows(ex, new, "pinn"), track, "extrapolation")
    out["extrapolation"] = len(new)

    wp = load_table(track, "sweep_physics_weight")
    new = sweep_physics_weight(
        prob, sorted(wp.w_phys.unique()), n_train=int(wp.n_train.iloc[0])
    )
    save_table(
        new[wp.columns.union(new.columns, sort=False)], track, "sweep_physics_weight"
    )
    out["sweep_physics_weight"] = len(new)
    return out


def rerun_headline(track: str, prob: Problem, seed: int = 11) -> None:
    """The headline pinn fit on all the data, as each track's run() makes it."""
    from physprior.io import load_json, save_json

    meta = load_json(track, "meta")
    f = fit_arm("pinn", prob, np.arange(len(prob)), seed=seed)
    meta["headline"]["pinn"] = {
        "params": f.params,
        "sigma": f.param_sigma,
        "expression": f.expression,
        "seconds": f.seconds,
    }
    meta["pinn_w_phys"] = prob.pinn_w_phys
    save_json(meta, track, "meta")
