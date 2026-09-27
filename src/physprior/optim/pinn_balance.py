"""What the PINN option `balance` does to the physics weight.

It was the frozen default until 2026-09-27; this measurement is why it no
longer is (see `methods/pinn.py`, FROZEN_PINN).

`balance` is Wang, Teng & Perdikaris (2021) learning-rate annealing, as
implemented in `methods.pinn._annealed_weight`: every `balance_every` epochs

    w <- alpha w + (1 - alpha) * max|grad_net L_data| / mean|grad_net L_phys|

For the shape-B PINN (y = law + sd_y NN, L_phys = mean(NN^2)) the physics
gradient is 2 mean(NN dNN/dtheta): it is proportional to the size of the
correction itself. The larger w gets, the smaller the correction, the
smaller the physics gradient, and the larger the next w. There is no fixed
point short of the correction vanishing; w grows until training ends. In
the Wang et al. setting the physics term is a PDE residual whose gradient
does not vanish with the network's output, so the scheme settles there. For a
penalty on the network's own output it does not.

This module measures that, on the real tracks, on the TUNING seeds, without
touching the frozen configuration:

    w_phys_final           the annealed weight at the end of training
    correction_rms_frac    RMS of NN over the training points, in units of
                           sd_y (the correction's size relative to the data)
    correction_rms_frac_out  the same over the held-out points: a correction
                           that is negligible where it was trained can still
                           be large where it extrapolates
    the same two per epoch, from the fit history
    the balanced and the unbalanced (w_phys = 1 fixed) PINN against the
    `physics` fit, in and out of range, and the distance between the
    balanced PINN's predictions and the physics fit's.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from physprior.benchmark.protocol import (
    fit_arm,
    score,
    split_extrapolate,
    split_random,
)
from physprior.methods import pinn as pinn_mod
from physprior.methods.base import TUNE_SEEDS

AREA = "optim"
TRACKS = ("gravity/kepler", "quantum/hydrogen", "quantum/cmb", "quantum/helium")
# The interpolation budgets each track's own w_phys sweep uses.
N_TRAIN = {
    "gravity/kepler": 6,
    "quantum/hydrogen": 25,
    "quantum/cmb": 30,
    "quantum/helium": 128,
}
VARIANTS = {
    "balanced": pinn_mod.PinnOptions(balance=True),
    "unbalanced": pinn_mod.DEFAULT_PINN,
}
RECORD_EVERY = 100


def load_problem(track: str):
    if track == "gravity/kepler":
        from physprior.problems.gravity import kepler as mod
    elif track == "quantum/hydrogen":
        from physprior.problems.quantum import hydrogen as mod  # type: ignore[no-redef]
    elif track == "quantum/cmb":
        from physprior.problems.quantum import cmb as mod  # type: ignore[no-redef]
    elif track == "quantum/helium":
        from physprior.problems.quantum import helium as mod  # type: ignore[no-redef]
    else:
        raise ValueError(f"unknown track {track!r}")
    prob, _ = mod.problem()
    return prob


def splits(track: str, prob, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The two splits each track's own study uses: a random interpolation
    split at the w_phys-sweep budget, and the track's extrapolation split."""
    n = len(prob)
    out = {"interp": split_random(n, min(N_TRAIN[track], n - 2), seed)}
    x0 = prob.x[:, 0]
    if track == "gravity/kepler":
        out["extrap"] = split_extrapolate(x0, 0.5)
    elif track == "quantum/hydrogen":
        out["extrap"] = split_extrapolate(x0, 0.25)
    elif track == "quantum/cmb":
        from physprior.problems.quantum.cmb import TURNOVER_HZ

        out["extrap"] = split_extrapolate(x0, float(np.mean(x0 < TURNOVER_HZ)))
    else:
        from physprior.problems.quantum.helium import extrapolation_split

        out["extrap"] = extrapolation_split(prob)
    return out


def _fit_pinn(prob, itr, seed, options, epochs=None):
    """The `pinn` arm exactly as `fit_arm` builds it, with the history on."""
    xtr, ytr = prob.sub(itr)
    return pinn_mod.fit_pinn(
        xtr,
        ytr,
        prob.law_t,
        prob.params,
        w_phys=1.0,
        seed=seed,
        epochs=epochs or prob.pinn_epochs,
        options=options,
        record_every=RECORD_EVERY,
    )


def history_frame(fit) -> pd.DataFrame:
    """Per-epoch weight and correction size, recovered from the history.

    The history stores data, phys and total = data + w phys, so
    w = (total - data) / phys; and phys = mean(NN^2), so the correction's
    RMS in units of sd_y is sqrt(phys).
    """
    h = fit.history or {}
    ep = np.asarray(h.get("epoch", []), float)
    data = np.asarray(h.get("data_loss", []), float)
    phys = np.asarray(h.get("phys_loss", []), float)
    total = np.asarray(h.get("loss", []), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(phys > 0, (total - data) / phys, np.nan)
    return pd.DataFrame(
        {
            "epoch": ep,
            "w_phys": w,
            "correction_rms_frac": np.sqrt(np.maximum(phys, 0.0)),
            "data_loss": data,
        }
    )


def run_cell(args) -> tuple[list[dict], list[pd.DataFrame]]:
    """One track, one seed: both splits, both variants, and `physics`."""
    import torch

    torch.set_num_threads(1)
    track, seed, epochs = args
    prob = load_problem(track)
    rows, hists = [], []
    scale = float(np.std(prob.y))
    for split_name, (itr, ite) in splits(track, prob, seed).items():
        phys = fit_arm("physics", prob, itr, seed)
        p_phys = phys.predict(prob.x[ite])
        base = score(prob, phys, itr, ite, split=split_name, seed=seed)
        rows.append({**base, "variant": "physics"})
        for vname, opts in VARIANTS.items():
            t0 = time.time()
            f = _fit_pinn(prob, itr, seed, opts, epochs)
            r = score(prob, f, itr, ite, split=split_name, seed=seed)
            gap = np.asarray(f.predict(prob.x[ite])) - np.asarray(p_phys)
            # The correction itself, out of range: prediction minus the law
            # at the PINN's own constants, in units of the training sd_y
            # (the scale the network's output is multiplied by).
            corr_out = np.asarray(f.predict(prob.x[ite])) - np.asarray(
                prob.law_np(prob.x[ite], **f.params)
            )
            sd_y = float(np.std(prob.y[itr])) or 1.0
            r.update(
                variant=vname,
                w_phys_final=f.extra["w_phys_final"],
                correction_rms_frac=f.extra["correction_rms_frac"],
                correction_rms_frac_out=float(np.sqrt(np.mean(corr_out**2)) / sd_y),
                data_mse_std_units=f.extra["data_mse_std_units"],
                rms_gap_to_physics_out=float(np.sqrt(np.mean(gap**2)) / scale),
                wall_seconds=time.time() - t0,
            )
            for k, v in f.params.items():
                r[f"dev_{k}_from_physics_ppm"] = (
                    (v - phys.params[k]) / phys.params[k] * 1e6
                )
            rows.append(r)
            h = history_frame(f)
            h.insert(0, "variant", vname)
            h.insert(0, "split", split_name)
            h.insert(0, "seed", seed)
            h.insert(0, "track", track)
            hists.append(h)
    return rows, hists


def run(
    quick: bool = False,
    tracks=TRACKS,
    seeds=TUNE_SEEDS,
    workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    from physprior.io import save_table

    epochs = 400 if quick else None
    seeds = seeds[:1] if quick else seeds
    workers = workers if workers is not None else (1 if quick else 3)
    cells = [(t, s, epochs) for t in tracks for s in seeds]
    if workers <= 1:
        res = [run_cell(c) for c in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(run_cell, cells))
    fits = pd.DataFrame([r for rows, _ in res for r in rows])
    hist = pd.concat([h for _, hs in res for h in hs], ignore_index=True)
    save_table(fits, AREA, "balance_fits")
    save_table(hist, AREA, "balance_history")
    summ = summarise(fits)
    save_table(summ, AREA, "balance_summary")
    slopes = feedback_slopes(hist)
    save_table(slopes, AREA, "balance_feedback")
    return {"fits": fits, "history": hist, "summary": summ, "feedback": slopes}


def summarise(fits: pd.DataFrame) -> pd.DataFrame:
    """Median over tuning seeds per (track, split, variant), with the ratio
    of each PINN variant's out-of-range error to the physics fit's."""
    g = (
        fits.groupby(["track", "split", "variant"])
        .agg(
            nrmse_in=("nrmse_in", "median"),
            nrmse_out=("nrmse_out", "median"),
            w_phys_final=("w_phys_final", "median"),
            correction_rms_frac=("correction_rms_frac", "median"),
            correction_rms_frac_out=("correction_rms_frac_out", "median"),
            rms_gap_to_physics_out=("rms_gap_to_physics_out", "median"),
        )
        .reset_index()
    )
    ref = g[g.variant == "physics"].set_index(["track", "split"])
    for col in ("nrmse_in", "nrmse_out"):
        g[f"{col}_vs_physics"] = [
            v / ref.loc[(t, s), col]
            for t, s, v in zip(g.track, g.split, g[col], strict=True)
        ]
    return g


def feedback_slopes(hist: pd.DataFrame) -> pd.DataFrame:
    """Log-log slope of w against correction size over the second half of
    training, per balanced run. A slope near -1 is the runaway described in
    the module docstring: w ~ 1 / |NN|, each feeding the other."""
    rows = []
    b = hist[hist.variant == "balanced"]
    for key, h in b.groupby(["track", "split", "seed"]):
        h = h[
            (h.epoch >= h.epoch.max() / 2)
            & (h.w_phys > 0)
            & (h.correction_rms_frac > 0)
        ]
        if len(h) < 4:
            continue
        slope = np.polyfit(np.log(h.correction_rms_frac), np.log(h.w_phys), 1)[0]
        growth = float(h.w_phys.iloc[-1] / h.w_phys.iloc[0])
        rows.append(
            {
                "track": key[0],
                "split": key[1],
                "seed": key[2],
                "slope_logw_vs_logcorr": float(slope),
                "w_growth_second_half": growth,
            }
        )
    return pd.DataFrame(rows)
