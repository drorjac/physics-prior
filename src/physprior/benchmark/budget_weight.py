"""H7: the `pinn` arm's physics weight, chosen per training-set size.

    physprior budget-weight

Each track's `pinn` arm has one physics weight for every budget
(`pinn_tuning`). Here the weight is chosen again for each budget of the
track's data-budget sweep, by the same rule on the same kind of data the
sweep scores: on the tuning seeds, a random quarter of that budget's training
points is held out for validation, candidates are scored by the median
validation nRMSE, ties go to the larger weight and a grid edge is extended.

The `pinn` arm is then re-scored on the reporting seeds with the per-budget
weight, on the same splits as the committed budget sweep, whose `physics`,
`nn` and single-weight `pinn` rows are read back for comparison. The
predictions this tests are in docs/HYPOTHESES.md, H7, written before the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.benchmark.metrics import nrmse
from physprior.benchmark.pinn_tuning import MAX_EXTEND, TIE, W_GRID
from physprior.benchmark.protocol import Problem, fit_arm, score, split_random
from physprior.io import save_table
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS

VAL_FRAC = 0.25


def _inner(n_train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    order = rng.permutation(n_train_idx)
    k = max(1, round(VAL_FRAC * len(order)))
    return np.sort(order[k:]), np.sort(order[:k])


def _score(prob: Problem, n_train: int, weights, seeds) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        itr, _ = split_random(len(prob), n_train, seed)
        fit_idx, val_idx = _inner(itr, seed)
        scale = float(np.std(prob.y[itr])) or 1.0
        for w in weights:
            f = fit_arm("pinn", prob, fit_idx, seed, w_phys=w)
            pred = np.asarray(f.predict(prob.x[val_idx]), float).ravel()
            rows.append(
                {
                    "track": prob.track,
                    "n_train": n_train,
                    "w_phys": w,
                    "seed": seed,
                    "val_nrmse": nrmse(prob.y[val_idx], pred, scale=scale),
                }
            )
    return pd.DataFrame(rows)


def _choose(df: pd.DataFrame) -> float:
    med = df.groupby("w_phys")["val_nrmse"].median()
    tied = med[med <= float(med.min()) * (1.0 + TIE)]
    return float(tied.index.max())


def select(prob: Problem, n_train: int, seeds=TUNE_SEEDS) -> tuple[float, pd.DataFrame]:
    df = _score(prob, n_train, W_GRID, seeds)
    chosen = _choose(df)
    for _ in range(MAX_EXTEND):
        lo, hi = float(df.w_phys.min()), float(df.w_phys.max())
        if chosen not in (lo, hi):
            break
        extra = lo / 10.0 if chosen == lo else hi * 10.0
        df = pd.concat([df, _score(prob, n_train, [extra], seeds)], ignore_index=True)
        chosen = _choose(df)
    df["chosen"] = df.w_phys == chosen
    df["pinned_at_edge"] = chosen in (float(df.w_phys.min()), float(df.w_phys.max()))
    return chosen, df


def reported(prob: Problem, budget_w: dict[int, float], arms=("pinn",)) -> pd.DataFrame:
    """The budget sweep's splits on the reporting seeds, with the per-budget
    weight for `pinn` and the track's defaults for any other arm asked for."""
    rows = []
    for nb, w in budget_w.items():
        for seed in REPORT_SEEDS:
            itr, ite = split_random(len(prob), nb, seed)
            for arm in arms:
                f = fit_arm(arm, prob, itr, seed, w_phys=w if arm == "pinn" else None)
                row = score(prob, f, itr, ite, sweep="budget", n_train=nb, seed=seed)
                row["w_phys"] = w if arm == "pinn" else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def run_track(
    track: str, prob: Problem, budgets, baseline: pd.DataFrame | None
) -> dict:
    """Choose a weight per budget, re-score, and write both tables.

    `baseline` is the committed budget sweep (physics, nn and single-weight
    pinn rows on the same splits). Where there is none, as for the simulated
    control, those arms are computed here.
    """
    chosen, frames = {}, []
    for nb in budgets:
        nb = int(min(nb, len(prob) - 2))
        w, df = select(prob, nb)
        chosen[nb] = w
        frames.append(df)
        print(f"  [{track}] n_train={nb:4d} w_phys={w:g}", flush=True)
    save_table(pd.concat(frames, ignore_index=True), track, "budget_w_phys_selection")

    new = reported(prob, chosen)
    new["arm"] = "pinn_budget_w"
    if baseline is None:
        single = reported(prob, dict.fromkeys(chosen, prob.pinn_w_phys or 1.0))
        others = reported(prob, chosen, arms=("physics", "nn"))
        baseline = pd.concat([single, others], ignore_index=True)
    keep = baseline[baseline.arm.isin(["physics", "nn", "pinn"])]
    table = pd.concat([keep, new], ignore_index=True)
    save_table(table, track, "budget_weight")
    return {"chosen": chosen}


# --------------------------------------------------------------------------
# the tracks
# --------------------------------------------------------------------------


def tracks() -> dict:
    """track -> () -> (Problem, budgets, committed budget sweep or None)."""
    from physprior.io import load_table

    def core(module_name: str, track: str):
        def make():
            import importlib

            prob, _ = importlib.import_module(module_name).problem()
            b = load_table(track, "sweep_budget")
            return prob, sorted(b.n_train.unique()), b

        return make

    def weather():
        from physprior.problems.fields import weather as W

        prob, _ = W.problem("july")
        b = load_table(W.track("july"), "sweep_budget")
        return prob, sorted(b.n_train.unique()), b

    def pulsar_control():
        from physprior.problems.gravity import pulsar_spindown as S

        sim = S.simulated_problem()
        sim.pinn_w_phys = S.PINN_W_PHYS_SIM
        return sim, [8, 12, 20, 30, 45], None

    return {
        "quantum/hydrogen": core(
            "physprior.problems.quantum.hydrogen", "quantum/hydrogen"
        ),
        "quantum/cmb": core("physprior.problems.quantum.cmb", "quantum/cmb"),
        "quantum/helium": core("physprior.problems.quantum.helium", "quantum/helium"),
        "fields/weather/july": weather,
        "gravity/pulsar_spindown/sim": pulsar_control,
    }


def run(only: str | None = None) -> dict:
    out = {}
    for track, make in tracks().items():
        if only and only != track:
            continue
        prob, budgets, baseline = make()
        out[track] = run_track(track, prob, budgets, baseline)
    return out


# --------------------------------------------------------------------------
# the verdicts, from the tables
# --------------------------------------------------------------------------


def verdicts() -> dict:
    from physprior.io import load_table

    p1, p2_cells, rows = {}, [], []
    for track in tracks():
        sel = load_table(track, "budget_w_phys_selection")
        w = sel[sel.chosen].groupby("n_train").w_phys.first()
        p1[track] = bool(w.iloc[0] >= w.iloc[-1])
        t = load_table(track, "budget_weight")
        med = t.groupby(["n_train", "arm"]).nrmse_out.median().unstack()
        for nb, r in med.iterrows():
            p2_cells.append(bool(r["pinn_budget_w"] <= r["pinn"]))
            rows.append(
                {
                    "track": track,
                    "n_train": int(nb),
                    "w_phys": float(w[nb]),
                    **r.to_dict(),
                }
            )
    n_ok = sum(p2_cells)
    return {
        "p1_weight_shrinks_or_holds": p1,
        "p1_failures": sum(not v for v in p1.values()),
        "p2_cells_no_worse": n_ok,
        "p2_cells": len(p2_cells),
        "refuted": (sum(not v for v in p1.values()) > 1)
        or (n_ok < 2 * len(p2_cells) / 3),
        "table": rows,
    }
