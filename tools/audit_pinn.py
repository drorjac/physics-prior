#!/usr/bin/env python3
"""Where does the `pinn` arm win, lose and tie, and by how much?

Reads `results/` and prints the tables in `docs/plans/PLAN.md` section 2. A gap
smaller than the pooled seed-to-seed spread is reported as a TIE rather than
a win: with three reporting seeds, a 2x difference on one metric is often
nothing at all.

    python tools/audit_pinn.py            # verdicts + the w_phys table

Nothing here selects anything -- it reads the REPORTING seeds and is a
description of the published results, not a tuning step.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from physprior.config import get_settings

TRACKS = ("gravity/kepler", "relativity/gw150914", "quantum/hydrogen", "quantum/cmb")
ARMS = ("oracle", "physics", "pinn", "sr", "nn")


def _verdict(pinn_mean, pinn_std, best_mean, best_std) -> str:
    pooled = float(np.hypot(pinn_std or 0.0, best_std or 0.0))
    if abs(pinn_mean - best_mean) <= pooled:
        return "TIE"
    return "PINN WINS" if pinn_mean < best_mean else "PINN LOSES"


def compare(results: pathlib.Path, track: str, fname: str, metric: str, label: str):
    """The pinn against the best arm that is not the oracle."""
    path = results / track / f"{fname}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if fname == "sweep_budget":
        df = df[df.n_train == df.n_train.max()]
    elif fname == "sweep_noise":
        df = df[df.noise_frac == df.noise_frac.max()]
    g = df.groupby("arm")[metric].agg(["mean", "std"])
    if "pinn" not in g.index:
        return None
    rivals = {a: g.loc[a, "mean"] for a in g.index if a not in ("pinn", "oracle")}
    best = min(rivals, key=rivals.get)
    pm, ps = g.loc["pinn", "mean"], g.loc["pinn", "std"]
    bm, bs = g.loc[best, "mean"], g.loc[best, "std"]
    return {
        "track": track,
        "question": label,
        "pinn": pm,
        "pinn_std": ps,
        "best_arm": best,
        "best": bm,
        "ratio": (pm / bm if bm else np.inf),
        "verdict": _verdict(pm, ps, bm, bs),
    }


def parameter_recovery(results: pathlib.Path, track: str):
    path = results / track / "sweep_budget.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df.n_train == df.n_train.max()]
    out = []
    for col in [c for c in df.columns if c.startswith("err_")]:
        g = df.groupby("arm")[col].mean().dropna()
        if "pinn" in g.index and "physics" in g.index:
            out.append((track, col, g["physics"], g["pinn"]))
    return out


def main() -> int:
    results = get_settings().results_dir
    rows = []
    for track in TRACKS:
        for fname, metric, label in (
            # nrmse_out is the HELD-OUT error in the budget and noise sweeps:
            # score() is called with the training indices as idx_in, so
            # nrmse_in there is the fit to the data the arm was handed.
            ("sweep_budget", "nrmse_out", "interpolation"),
            ("extrapolation", "nrmse_out", "extrapolation"),
            ("sweep_noise", "nrmse_out", "noise (max)"),
        ):
            r = compare(results, track, fname, metric, label)
            if r:
                rows.append(r)
    if not rows:
        print("no results yet -- run `make run`")
        return 1

    print("PINN vs the best non-oracle arm, reporting seeds\n")
    print(
        f"{'track':22s} {'question':14s} {'pinn':>11s} {'best':>11s} "
        f"{'arm':>8s} {'ratio':>9s}  verdict"
    )
    for r in rows:
        print(
            f"{r['track']:22s} {r['question']:14s} {r['pinn']:11.3g} "
            f"{r['best']:11.3g} {r['best_arm']:>8s} {r['ratio']:9.3g}x  "
            f"{r['verdict']}"
        )

    tally = pd.Series([r["verdict"] for r in rows]).value_counts().to_dict()
    print("\nscore:", ", ".join(f"{v} {k}" for k, v in tally.items()))

    print("\nparameter recovery (mean % error at the largest budget)\n")
    print(f"{'track':22s} {'quantity':16s} {'physics':>12s} {'pinn':>12s}  better")
    for track in TRACKS:
        for tr, col, phys, pinn in parameter_recovery(results, track):
            better = "physics" if abs(phys) <= abs(pinn) else "pinn"
            print(f"{tr:22s} {col:16s} {phys:12.4g} {pinn:12.4g}  {better}")

    print("\nw_phys sweep -- REPORTING seeds, so it may not be used to pick a")
    print("default (invariant 6). Shown because it bounds what is on the table.\n")
    for track in TRACKS:
        path = results / track / "sweep_physics_weight.csv"
        if not path.exists():
            continue
        g = pd.read_csv(path).groupby("w_phys")["nrmse_out"].median()
        cur, best_w = g.get(1.0, np.nan), g.idxmin()
        print(
            f"{track:22s} w=1.0 -> {cur:.3g}   best w={best_w:<8g} -> "
            f"{g.min():.3g}   ({cur / g.min():.1f}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
