"""Track gravity/pulsar_spindown -- the braking index of young pulsars.

DATA     n_obs = nu nu_ddot / nu_dot^2 for the ATNF catalogue's isolated,
         non-magnetar pulsars younger than 1e4 yr with nu_ddot at 5 sigma or
         better (`data/sources/atnf.py`, rule fixed before any fit).

LAW      Magnetic dipole in vacuum:  nu_dot = -K nu^n  with  n = 3, so the
         braking index of every pulsar is the same constant:

             n_obs = n

PARAM    n, published as 3 (Pacini 1968; Gunn & Ostriker 1969).

INPUTS   x = (log10 characteristic age [yr], log10 surface field [G]), so a
         learned correction can use both. The law ignores them.

WHY THIS TRACK   The law is incomplete, as on helium, but here what it
         misses is not expected to be a function of the inputs: indices
         below 3 are pulsar-specific torque physics, and the older pulsars'
         indices are dominated by glitch recovery. The refined H3 predicts a
         correction cannot help; a simulated control, where the index does
         depend on age through a known term, checks that the method could
         have learned such a dependence if it existed. The predictions are in
         docs/HYPOTHESES.md, written before these runs.

EXTRAPOLATION    train on the younger half, predict the older half.

NOT FITTED       the target is dimensionless and the pulsar-to-pulsar
         scatter exceeds the quoted errors by orders of magnitude, so the
         fits are unweighted, as on helium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from physprior.benchmark.protocol import (
    Problem,
    fit_arm,
    law,
    split_extrapolate,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import BRAKING_INDEX_DIPOLE
from physprior.data.sources import atnf
from physprior.io import save_json, save_table
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS
from physprior.methods.pinn import PhysParam

TRACK = "gravity/pulsar_spindown"
SIM_TRACK = "gravity/pulsar_spindown/sim"
TRAIN_FRAC = 0.5  # the younger half

# The `pinn` arm's physics weight, chosen on the tuning seeds by
# `physprior.benchmark.pinn_tuning` (results/gravity/pulsar_spindown/tune/).
PINN_W_PHYS = 1e6  # pinned at the top of the grid: the validation block
#                    prefers no correction, so the arm is the law by choice
# The control's own weight, by the same rule on its own training half
# (results/gravity/pulsar_spindown/sim/tune/).
PINN_W_PHYS_SIM = 1e-3
# Chosen by `tune_nn()` on the tuning seeds, on a held-out half of the
# younger pulsars, before any reporting seed was run (validation RMSE 0.72).
NN_CFG = dict(width=32, depth=3, weight_decay=1e-5, epochs=4000)

# The simulated control: n depends on age through a known term, so a
# correction has something real to learn. Pulsar ages and fields are drawn
# over the real sample's range.
SIM_N = 60
SIM_SLOPE = -0.8  # d n / d log10(age)
SIM_PIVOT = 3.0  # log10(age / yr) where n = 3
SIM_NOISE = 0.05  # measurement scatter on n_obs


@law("n")
def law_np(x, n):
    return n * np.ones(np.atleast_2d(x).shape[0])


def law_t(x, n):
    return n * torch.ones(x.shape[0], dtype=x.dtype)


def _problem(track: str, x: np.ndarray, y: np.ndarray, notes: str) -> Problem:
    return Problem(
        track=track,
        x=x,
        y=y,
        law_np=law_np,
        law_t=law_t,
        params=[PhysParam("n", BRAKING_INDEX_DIPOLE, positive=True, lo=0.1, hi=100.0)],
        theta_published={"n": BRAKING_INDEX_DIPOLE},
        xlabel="log10 characteristic age [yr]",
        ylabel="braking index n",
        sigma=None,
        sr_kwargs=dict(
            feature_names=["log_age", "log_b"],
            niterations=40,
            maxsize=12,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
        ),
        nn_cfg=dict(NN_CFG),
        pinn_epochs=4000,
        pinn_w_phys=PINN_W_PHYS,
        notes=notes,
    )


def problem() -> tuple[Problem, dict]:
    p = atnf.load()
    df = p.frame
    x = np.column_stack([np.log10(df.age_yr), np.log10(df.b_surface_g)])
    prob = _problem(
        TRACK,
        x,
        df.n_obs.to_numpy(),
        f"ATNF v{p.catalogue_version}: {len(df)} young isolated pulsars",
    )
    meta = {
        "provenance": p.provenance,
        "catalogue_version": p.catalogue_version,
        "n_catalogue": p.n_catalogue,
        "n_with_nu_ddot": p.n_with_nu_ddot,
        "n_selected": len(df),
        "selection": atnf.SELECTION,
        "pulsars": df[["psrj", "age_yr", "b_surface_g", "n_obs", "n_obs_err"]]
        .round(6)
        .to_dict(orient="records"),
        "weighting": "unweighted: pulsar-to-pulsar scatter exceeds the quoted errors",
    }
    return prob, meta


def simulated_problem(
    seed: int = 0, lo: np.ndarray | None = None, hi: np.ndarray | None = None
) -> Problem:
    """The control: n_obs = 3 + SIM_SLOPE (log age - SIM_PIVOT) + noise.

    Ages and fields are drawn uniformly over the real sample's range, so the
    control asks the same extrapolation question with a learnable answer.
    """
    if lo is None or hi is None:
        real, _ = problem()
        lo, hi = real.x.min(axis=0), real.x.max(axis=0)
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(SIM_N, 2))
    n_true = BRAKING_INDEX_DIPOLE + SIM_SLOPE * (x[:, 0] - SIM_PIVOT)
    y = n_true + rng.normal(0.0, SIM_NOISE, SIM_N)
    return _problem(SIM_TRACK, x, y, "simulated: n falls linearly with log age")


def train_split(prob: Problem) -> np.ndarray:
    itr, _ = split_extrapolate(prob.x[:, 0], TRAIN_FRAC)
    return itr


# --------------------------------------------------------------------------
# the black box's configuration, chosen on the tuning seeds
# --------------------------------------------------------------------------


def tune_nn(epochs: int = 4000) -> dict:
    from physprior.benchmark.protocol import split_random
    from physprior.methods.neural import tune_mlp

    prob, _ = problem()
    itr = train_split(prob)
    fit_idx, val_idx = split_random(len(itr), len(itr) // 2, seed=TUNE_SEEDS[0])
    a, b = itr[fit_idx], itr[val_idx]
    return tune_mlp(
        prob.x[a], prob.y[a], prob.x[b], prob.y[b], TUNE_SEEDS, epochs=epochs
    )


# --------------------------------------------------------------------------
# the predictions
# --------------------------------------------------------------------------


def dipole_test(prob: Problem) -> dict:
    """Prediction 1: the fitted n on the training half, against 3."""
    itr = train_split(prob)
    f = fit_arm("physics", prob, itr, seed=REPORT_SEEDS[0])
    n, sig = f.params["n"], f.param_sigma["n"]
    return {
        "n_fit": float(n),
        "n_sigma": float(sig),
        "n_train": len(itr),
        "deficit_sigma": float((BRAKING_INDEX_DIPOLE - n) / sig),
        "below_three": bool(BRAKING_INDEX_DIPOLE - n > sig),
    }


def verdicts(ex: pd.DataFrame, sim_ex: pd.DataFrame, dipole: dict) -> dict:
    """Each prediction against its criterion, from the tables."""

    def ratio(df):
        w = df.pivot_table(index="seed", columns="arm", values="nrmse_out")
        return w["pinn"] / w["physics"]

    real, sim = ratio(ex), ratio(sim_ex)
    arms = ex[ex.arm != "oracle"].groupby("arm")["nrmse_out"].median()
    return {
        "p1_dipole_law_incomplete": dipole["below_three"],
        "p2_pinn_not_better_on_every_seed": bool(not (real < 1).all()),
        "p3_every_arm_fails_out_of_range": bool((arms > 1).all()),
        "p4_pinn_better_in_control_on_every_seed": bool((sim < 1).all()),
        "pinn_over_physics_real": real.round(4).to_dict(),
        "pinn_over_physics_sim": sim.round(4).to_dict(),
        "median_nrmse_out": arms.round(4).to_dict(),
    }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def figure(prob: Problem, sim: Problem) -> None:
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    itr = train_split(prob)
    fit = fit_arm("physics", prob, itr, seed=REPORT_SEEDS[0])
    ax = axes[0]
    young = np.isin(np.arange(len(prob)), itr)
    ax.scatter(prob.x[young, 0], prob.y[young], color="#2a78d6", label="training half")
    ax.scatter(prob.x[~young, 0], prob.y[~young], color="#eb6834", label="older half")
    ax.axhline(BRAKING_INDEX_DIPOLE, color="#52514e", ls="--", label="dipole, n = 3")
    ax.axhline(fit.params["n"], color="#1baf7a", label="n fitted on training half")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlabel(prob.xlabel)
    ax.set_ylabel(prob.ylabel)
    ax.set_title("ATNF pulsars")
    ax.legend(fontsize=8)

    ax = axes[1]
    sitr = train_split(sim)
    syoung = np.isin(np.arange(len(sim)), sitr)
    ax.scatter(sim.x[syoung, 0], sim.y[syoung], color="#2a78d6", s=12)
    ax.scatter(sim.x[~syoung, 0], sim.y[~syoung], color="#eb6834", s=12)
    # predictions at the held-out pulsars themselves: the network depends on
    # the field as well as the age, so a curve at one field would misdraw it
    old = ~syoung
    phys = fit_arm("physics", sim, sitr, seed=REPORT_SEEDS[0])
    ax.axhline(phys.params["n"], color="#1baf7a", label="physics (constant n)")
    pinn = fit_arm("pinn", sim, sitr, seed=REPORT_SEEDS[0])
    ax.scatter(
        sim.x[old, 0],
        np.asarray(pinn.predict(sim.x[old])).ravel(),
        marker="x",
        color="#4a3aa7",
        s=18,
        label="pinn, out of range",
    )
    ax.set_xlabel(sim.xlabel)
    ax.set_title("simulated control: n falls with age")
    ax.legend(fontsize=8)
    P.save(fig, "gravity", "pulsar_braking_index")


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    print(f"[{TRACK}] {len(prob)} pulsars", flush=True)

    budgets = [4, 6, 8, 10] if not quick else [6]
    noise = [0.0, 0.01, 0.05, 0.15] if not quick else [0.0]
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [1.0]

    save_table(sweep_budget(prob, budgets), TRACK, "sweep_budget")
    save_table(sweep_noise(prob, noise, n_train=9), TRACK, "sweep_noise")
    ex = study_extrapolation(prob, train_frac=TRAIN_FRAC)
    save_table(ex, TRACK, "extrapolation")
    save_table(
        sweep_physics_weight(prob, weights, n_train=9), TRACK, "sweep_physics_weight"
    )

    sim = simulated_problem()
    sim.pinn_w_phys = PINN_W_PHYS_SIM
    sim_ex = study_extrapolation(sim, train_frac=TRAIN_FRAC)
    save_table(sim_ex, SIM_TRACK, "extrapolation")

    meta["dipole_test"] = dipole_test(prob)
    meta["verdicts"] = verdicts(ex, sim_ex, meta["dipole_test"])
    meta["simulation"] = {
        "n": SIM_N,
        "slope": SIM_SLOPE,
        "pivot": SIM_PIVOT,
        "noise": SIM_NOISE,
        "pinn_w_phys": PINN_W_PHYS_SIM,
    }

    head = {}
    for arm in ("physics", "pinn", "sr"):
        f = fit_arm(arm, prob, np.arange(len(prob)), seed=REPORT_SEEDS[0])
        head[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "expression": f.expression,
            "seconds": f.seconds,
        }
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    figure(prob, sim)
    return meta
