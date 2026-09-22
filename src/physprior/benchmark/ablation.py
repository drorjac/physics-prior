"""Decide the `pinn` arm's defaults on the TUNING seeds, and nowhere else.

    physprior tune            # every option, every track
    physprior tune --quick    # shorter training, for a smoke test

Invariant 6 is the whole design of this module. Every measurement here runs
on seeds 3 / 7 / 19 and is written under `results/<track>/tune/`, separate
from the reported sweeps on 11 / 23 / 42, so that turning an option on can
never be a decision made by looking at a reported number.

What it takes to ship (PLAN.md section 5.2): an option goes into the default
configuration only if, on the tuning seeds, it helps on at least one track,
and on no track does it improve in-distribution error while degrading
parameter recovery. The second half is the one that matters -- a correction
network that fits better while the constant drifts has taken the physics out
of the physics-informed arm, which is finding 4 of this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.benchmark.metrics import nrmse
from physprior.benchmark.protocol import (
    Problem,
    fit_arm,
    split_extrapolate,
    split_random,
)
from physprior.config import get_settings
from physprior.methods.base import TUNE_SEEDS
from physprior.methods.pinn import PinnOptions

# One switch at a time. A stack of them is a different experiment, and the
# ablation has to say which change did the work.
ABLATIONS: dict[str, PinnOptions] = {
    "baseline": PinnOptions(),
    "early": PinnOptions(early_stopping=True),
    "lbfgs": PinnOptions(lbfgs=True),
    "ens5": PinnOptions(ensemble=5),
    "fourier": PinnOptions(fourier=16),
    "balance": PinnOptions(balance=True),
    # The stack. Two options that each earn their place separately are still
    # a configuration nobody measured; gradient-norm balancing changes the
    # loss every member of an ensemble is trained on, so the two are not
    # obviously independent and the combination is ablated in its own right.
    "balance+ens5": PinnOptions(balance=True, ensemble=5),
}


def _problems() -> dict:
    """track -> () -> (Problem, meta). Imported lazily: the data layer
    touches the disk cache, and a caller may only want one track."""
    from physprior.problems.gravity import kepler
    from physprior.problems.quantum import cmb, hydrogen
    from physprior.problems.relativity import gw150914

    return {
        "gravity/kepler": kepler.problem,
        "relativity/gw150914": gw150914.problem,
        "quantum/hydrogen": hydrogen.problem,
        "quantum/cmb": cmb.problem,
    }


def _primary_param(prob: Problem) -> str | None:
    for spec in prob.params:
        if spec.name in prob.theta_published:
            return spec.name
    return None


def _score_one(prob: Problem, opts: PinnOptions, seed: int, quick: bool) -> list:
    """The three questions an option can plausibly move, on one seed."""
    rows = []
    name = _primary_param(prob)
    published = prob.theta_published.get(name) if name else None
    epochs = max(400, prob.pinn_epochs // (4 if quick else 1))
    original, prob.pinn_epochs = prob.pinn_epochs, epochs

    try:
        n_train = max(int(0.7 * len(prob)), 4)
        itr, ite = split_random(len(prob), n_train, seed)
        fit = fit_arm("pinn", prob, itr, seed, pinn_options=opts)
        scale = float(np.std(prob.y))
        err = None
        if name and published:
            err = abs(fit.params.get(name, np.nan) - published) / abs(published) * 100
        rows.append(
            {
                "track": prob.track,
                "option": opts.tag,
                "seed": seed,
                "question": "interpolation",
                "nrmse": nrmse(prob.y[ite], fit.predict(prob.x[ite]), scale=scale),
                "param_err_pct": err,
                "param": name,
                "seconds": fit.seconds,
                "engaged": bool(fit.extra.get("engaged", True)),
            }
        )

        jtr, jte = split_extrapolate(prob.x[:, 0], 0.5)
        efit = fit_arm("pinn", prob, jtr, seed, pinn_options=opts)
        rows.append(
            {
                "track": prob.track,
                "option": opts.tag,
                "seed": seed,
                "question": "extrapolation",
                "nrmse": nrmse(prob.y[jte], efit.predict(prob.x[jte]), scale=scale),
                "param_err_pct": (
                    abs(efit.params.get(name, np.nan) - published)
                    / abs(published)
                    * 100
                    if name and published
                    else None
                ),
                "param": name,
                "seconds": efit.seconds,
                "engaged": bool(efit.extra.get("engaged", True)),
            }
        )
    finally:
        prob.pinn_epochs = original
    return rows


def run(
    tracks=None, options=None, seeds=TUNE_SEEDS, quick: bool = False
) -> pd.DataFrame:
    """Every option on every track, on the tuning seeds."""
    probs = _problems()
    tracks = list(tracks or probs)
    options = options or ABLATIONS
    rows = []
    for track in tracks:
        prob, _ = probs[track]()
        for tag, opts in options.items():
            for seed in seeds:
                rows.append(_score_one(prob, opts, seed, quick))
                print(f"  {track:22s} {tag:10s} seed={seed} done", flush=True)
    flat = [r for group in rows for r in group]
    return pd.DataFrame(flat)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Each option against the baseline, per track and question.

    `ratio` below 1 means the option helped. It is a ratio of medians over
    the tuning seeds, because a mean over three seeds is hostage to one bad
    initialisation.
    """
    agg: dict = {"nrmse": "median", "param_err_pct": "median"}
    if "engaged" in df.columns:
        agg["engaged"] = "any"
    g = df.groupby(["track", "question", "option"]).agg(agg).reset_index()
    base = g[g.option == "baseline"].set_index(["track", "question"])
    out = []
    for _, r in g[g.option != "baseline"].iterrows():
        key = (r["track"], r["question"])
        if key not in base.index:
            continue
        b = base.loc[key]
        out.append(
            {
                "track": r["track"],
                "question": r["question"],
                "option": r["option"],
                "baseline_nrmse": b["nrmse"],
                "option_nrmse": r["nrmse"],
                "ratio": r["nrmse"] / b["nrmse"] if b["nrmse"] else np.nan,
                "baseline_param_err": b["param_err_pct"],
                "option_param_err": r["param_err_pct"],
                "param_ratio": (
                    r["param_err_pct"] / b["param_err_pct"]
                    if b["param_err_pct"]
                    else np.nan
                ),
                "engaged": bool(r.get("engaged", True)),
            }
        )
    return pd.DataFrame(out)


# An option has to beat the baseline by more than this to count as helping;
# below it the difference is not worth changing a default for.
HELP = 0.9
HURT = 1.1


def decide(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the shipping rule to the ablation table.

    The disqualifying pattern is explicit: an option that improves
    in-distribution error while the recovered constant gets worse has bought
    accuracy with the physics, and does not ship however good the nRMSE is.
    """
    rows = []
    for option, sub in summary.groupby("option"):
        if "engaged" in sub.columns and not sub["engaged"].any():
            rows.append(
                {
                    "option": option,
                    "tracks helped": 0,
                    "tracks hurt": 0,
                    "ship": "n/a",
                    "why": (
                        "never engaged on any track -- e.g. early stopping "
                        "needs a training set big enough to carve a "
                        "validation split from, and L-BFGS is reverted when "
                        "its proposal does not lower the loss"
                    ),
                }
            )
            continue
        helped = sub[sub.ratio < HELP]
        hurt = sub[sub.ratio > HURT]
        bought = sub[(sub.ratio < HELP) & (sub.param_ratio > HURT)]
        # An option that helps one track and hurts five is not a default.
        # PLAN.md 5.2 only said "helps on at least one track", which the
        # Fourier-feature run then satisfied while making three tracks up to
        # 98x worse -- the rule had simply never considered an option that
        # hurts. A default has to be safe everywhere it is on.
        ship = bool(len(helped)) and not len(bought) and not len(hurt)
        why = []
        if len(helped):
            why.append(
                "helps on "
                + ", ".join(f"{r.track} {r.question}" for r in helped.itertuples())
            )
        else:
            why.append("no track improved by more than 10%")
        if len(hurt):
            why.append(
                "hurts on "
                + ", ".join(f"{r.track} {r.question}" for r in hurt.itertuples())
            )
        if len(bought):
            why.append(
                "DISQUALIFIED: better fit but worse constant on "
                + ", ".join(f"{r.track} {r.question}" for r in bought.itertuples())
            )
        rows.append(
            {
                "option": option,
                "tracks helped": len(helped),
                "tracks hurt": len(hurt),
                "ship": "yes" if ship else "no",
                "why": "; ".join(why),
            }
        )
    return pd.DataFrame(rows).sort_values("option").reset_index(drop=True)


def tune_physics_weight(
    weights=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 1000.0), tracks=None, quick: bool = False
) -> pd.DataFrame:
    """The `w_phys` sweep again, on the TUNING seeds, so its answer may be
    used to set a default. The reported sweep on 11/23/42 stays untouched."""
    from physprior.benchmark.protocol import sweep_physics_weight

    probs = _problems()
    frames = []
    for track in list(tracks or probs):
        prob, _ = probs[track]()
        if quick:
            prob.pinn_epochs = max(400, prob.pinn_epochs // 4)
        df = sweep_physics_weight(prob, weights, seeds=TUNE_SEEDS)
        df["track"] = track
        frames.append(df)
        print(f"  {track} w_phys sweep done", flush=True)
    return pd.concat(frames, ignore_index=True)


def tune_dir(track: str):
    d = get_settings().results_dir / track / "tune"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(df: pd.DataFrame, name: str) -> None:
    for track, sub in df.groupby("track"):
        path = tune_dir(str(track)) / f"{name}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}")
