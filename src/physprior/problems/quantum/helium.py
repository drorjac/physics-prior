"""Track quantum/helium -- NIST He I term energies. The law is incomplete.

DATA     E(n, l, S), the singly excited terms 1s.nl of He I above the 1s2
         ground state, n = 2..35, l = 0..7, singlet and triplet: 452 terms.
         NIST evaluated reference values (see data/sources/nist.py).

LAW      Hydrogenic:   E = L - R / n^2

PARAM    L, the ionisation energy, and R, the Rydberg constant for He.

WHY THIS TRACK   Hydrogen's law is complete for one electron. Helium has two:
         the inner electron screens the nucleus, and an outer electron that
         penetrates the core sees more of its charge. Each series is shifted
         by a quantum defect,  E = L - R / (n - d_ls)^2,  large for S (0.14
         singlet, 0.30 triplet), small for P, and near zero from F upwards.
         Expanded, the defect adds a term going as 1/n^3 -- a different
         shape from the law's 1/n^2, and one that depends on l and S, which
         the law does not see. This is the case H3 predicts a learned
         correction should win: the missing physics is distinguishable from
         the law (docs/HYPOTHESES.md, H3).

EXTRAPOLATION    train on n <= 10 (102 terms), predict n = 11..35 (350).

COMPLETE LAW     The Rydberg-Ritz form with a per-series defect,
         d_ls(n) = d0 + d2 / (n - d0)^2, is fitted on the same split and
         reported beside the arms, as Mercury's complete model is. It is
         not an arm: it knows the answer's form.

H3 PREDICTIONS   1. pinn beats the hydrogenic `physics` fit out of range;
         2. pinn does not beat the Rydberg-Ritz fit;
         3. the pinn's predictions carry a quantum defect with the data's l
         structure: large for S, near zero for high l.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from physprior.benchmark.metrics import nrmse
from physprior.benchmark.protocol import (
    Problem,
    fit_arm,
    law,
    split_extrapolate,
    split_random,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import HE_I_IONISATION_CM, RYDBERG_HE_CM
from physprior.data.sources import nist
from physprior.io import save_json, save_table
from physprior.methods import pinn as pinn_mod
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS, Fit
from physprior.methods.pinn import PhysParam
from physprior.util import first_column

TRACK = "quantum/helium"
SR_SCALE = 1.0e5  # cm^-1 -> O(1), so PySR's constants stay sane
N_TRAIN_MAX = 10  # extrapolation: fit n <= 10, predict n = 11..35
L_LETTERS = "SPDFGHIK"

# Chosen by `tune_nn()` on the tuning seeds (3/7/19), on a held-out half of
# the n <= 10 terms, before any reporting seed was run (validation RMSE
# 335.5 cm^-1). Re-run it to check.
NN_CFG = dict(width=32, depth=3, weight_decay=1e-5, epochs=6000)


@law("L - R/n**2")
def law_np(x, L, R):
    n = first_column(x)
    return L - R / n**2


def law_t(x, L, R):
    n = x[:, 0] if x.ndim > 1 else x
    return L - R / n**2


def _sr_transform(x, y):
    """SR sees (n, l, s) and E/1e5; the back-transform undoes the scaling."""
    X = np.asarray(x, float)

    def back(xq, raw_predict):
        return raw_predict(np.asarray(xq, float)) * SR_SCALE

    return X, np.asarray(y, float) / SR_SCALE, back


def problem() -> tuple[Problem, dict]:
    he = nist.load_helium()
    x = np.column_stack([he.n, he.l, he.s])
    prob = Problem(
        track=TRACK,
        x=x,
        y=he.energy_icm,
        law_np=law_np,
        law_t=law_t,
        params=[
            PhysParam("L", 2.0e5, positive=True, lo=1.5e5, hi=2.5e5),
            PhysParam("R", 1.0e5, positive=True, lo=1.0e4, hi=1.0e6),
        ],
        theta_published={"L": HE_I_IONISATION_CM, "R": RYDBERG_HE_CM},
        xlabel="principal quantum number n",
        ylabel="E  [cm$^{-1}$]",
        # Unweighted on purpose. The quoted uncertainties run from 2e-9 to
        # 7e-5 cm^-1, while the hydrogenic law misses by up to thousands of
        # cm^-1. Weighting by them would let the few most precisely tabulated
        # terms decide the fit; unweighted, curve_fit scales the parameter
        # errors by the actual residuals, which is right for a law known to
        # be incomplete.
        sigma=None,
        sr_transform=_sr_transform,
        sr_kwargs=dict(
            feature_names=["n", "l", "s"],
            niterations=60,
            maxsize=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "inv"],
        ),
        nn_cfg=dict(NN_CFG),
        notes="NIST ASD He I singly excited terms 1s.nl, n=2..35, l=0..7",
    )
    meta = {
        "provenance": he.provenance,
        "n_terms": len(he),
        "ionisation_limit_icm": he.limit_icm,
        "ionisation_limit_sigma_icm": he.limit_sigma_icm,
        "rydberg_he_icm": RYDBERG_HE_CM,
        "weighting": "unweighted: model error exceeds the quoted uncertainties",
    }
    return prob, meta


def extrapolation_split(prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    """n <= 10 against n > 10, exactly, whatever the tie order in n."""
    frac = float(np.mean(prob.x[:, 0] <= N_TRAIN_MAX))
    itr, ite = split_extrapolate(prob.x[:, 0], frac)
    assert prob.x[itr, 0].max() <= N_TRAIN_MAX < prob.x[ite, 0].min()
    return itr, ite


# --------------------------------------------------------------------------
# The black box's configuration, chosen on the tuning seeds
# --------------------------------------------------------------------------


def tune_nn(epochs: int = 4000) -> dict:
    """Grid-search the MLP on a held-out half of the n <= 10 terms.

    Only the extrapolation training range is used, and only the tuning
    seeds, so no reported number can have influenced the choice.
    """
    from physprior.methods.neural import tune_mlp

    prob, _ = problem()
    itr, _ = extrapolation_split(prob)
    fit_idx, val_idx = split_random(len(itr), len(itr) // 2, seed=TUNE_SEEDS[0])
    a, b = itr[fit_idx], itr[val_idx]
    return tune_mlp(
        prob.x[a], prob.y[a], prob.x[b], prob.y[b], TUNE_SEEDS, epochs=epochs
    )


# --------------------------------------------------------------------------
# The complete law, for comparison
# --------------------------------------------------------------------------


def _series(x: np.ndarray) -> list[tuple[int, int]]:
    return sorted({(int(l), int(s)) for l, s in x[:, 1:3]})


def fit_rydberg_ritz(x: np.ndarray, y: np.ndarray) -> Fit:
    """E = L - R_He / (n - d)^2, with d = d0 + d2 / (n - d0)^2 per series.

    R is held at the reduced-mass value for He; L and the two defect
    coefficients of every (l, S) series are fitted. A series with fewer than
    three training terms gets d2 = 0.
    """
    t0 = time.time()
    series = _series(x)
    index = {k: i for i, k in enumerate(series)}
    counts = {k: int(np.sum((x[:, 1] == k[0]) & (x[:, 2] == k[1]))) for k in series}

    def unpack(p):
        return p[0], p[1::2], p[2::2]

    def model(p, xq):
        L, d0, d2 = unpack(p)
        k = np.array([index.get((int(l), int(s)), -1) for l, s in xq[:, 1:3]])
        if np.any(k < 0):
            raise ValueError("a term belongs to a series with no training data")
        n = xq[:, 0]
        n0 = n - d0[k]
        return L - RYDBERG_HE_CM / (n - d0[k] - d2[k] / n0**2) ** 2

    def resid(p):
        r = model(p, x) - y
        _, _, d2 = unpack(p)
        # series with too few points: pin d2 to zero rather than let it float
        pin = [d2[index[k]] for k in series if counts[k] < 3]
        return np.concatenate([r, 1e3 * np.asarray(pin, float)])

    p0 = np.concatenate([[float(y.max()) + 100.0], np.zeros(2 * len(series))])
    sol = least_squares(resid, p0, x_scale="jac", max_nfev=20000)
    L, d0, d2 = unpack(sol.x)
    return Fit(
        name="ritz",
        predict=lambda xq: model(sol.x, np.asarray(xq, float)),
        params={"L": float(L)},
        expression="L - R_He/(n - d_ls)^2,  d_ls = d0 + d2/(n - d0)^2",
        n_free=1 + 2 * len(series),
        seconds=time.time() - t0,
        extra={
            "defects": {
                f"{L_LETTERS[l]}{2 * s + 1}": {"d0": float(d0[i]), "d2": float(d2[i])}
                for (l, s), i in index.items()
            }
        },
    )


# --------------------------------------------------------------------------
# What was learned: the quantum defect each arm's predictions imply
# --------------------------------------------------------------------------


def implied_defect(x: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """d_eff = n - sqrt(R_He / (L - E)), with L and R the reference values.

    The reference is the same for every arm, so d_eff is a property of the
    predicted energies alone. The hydrogenic law, whatever its fitted
    constants, gives the same d_eff for every l; a correction that learned
    the physics gives an S series that stands apart.
    """
    gap = HE_I_IONISATION_CM - np.asarray(energy, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return x[:, 0] - np.sqrt(RYDBERG_HE_CM / gap)


def defect_table(prob: Problem, fits: dict[str, Fit], idx: np.ndarray) -> pd.DataFrame:
    """Median implied defect per (l, S) series over the terms `idx`.

    Series are comparable only over matched n ranges: with constants that are
    not the reference ones, a law blind to l still gives an implied defect
    that drifts with n. On the n > 10 test split every series from S to I
    covers n = 11..35, so the comparison there is like for like.
    """
    rows = []
    xq = prob.x[idx]
    preds = {"data": prob.y[idx]} | {
        name: np.asarray(f.predict(xq), float).ravel() for name, f in fits.items()
    }
    for l, s in _series(xq):
        sel = (xq[:, 1] == l) & (xq[:, 2] == s)
        row = {
            "series": f"{L_LETTERS[l]}{2 * s + 1}",
            "l": l,
            "s": s,
            "n_terms": int(sel.sum()),
        }
        for name, e in preds.items():
            row[name] = float(np.nanmedian(implied_defect(xq[sel], e[sel])))
        rows.append(row)
    return pd.DataFrame(rows)


def defect_structure(table: pd.DataFrame, arm: str) -> dict:
    """Does `arm` reproduce the data's l structure?

    Two numbers: the correlation of the arm's per-series defect with the
    data's, and the S-series defect minus the mean defect from F upwards,
    for the arm and for the data. Prediction 3 holds if the correlation is
    high and the S-minus-high-l gap has the data's sign and size.
    """
    t = table.dropna(subset=[arm, "data"])
    high = t[t.l >= 3]
    s_rows = t[t.l == 0]
    gap = lambda col: float(s_rows[col].mean() - high[col].mean())  # noqa: E731
    return {
        "arm": arm,
        "corr_with_data": float(np.corrcoef(t[arm], t["data"])[0, 1]),
        "s_minus_high_l_arm": gap(arm),
        "s_minus_high_l_data": gap("data"),
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _score_out(prob, fit, itr, ite) -> dict:
    scale = float(np.std(prob.y))
    return {
        "nrmse_in": nrmse(prob.y[itr], fit.predict(prob.x[itr]), scale=scale),
        "nrmse_out": nrmse(prob.y[ite], fit.predict(prob.x[ite]), scale=scale),
    }


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    n = len(prob)
    print(
        f"[{TRACK}] {n} terms, n = {prob.x[:, 0].min():.0f}..{prob.x[:, 0].max():.0f}"
    )

    budgets = [8, 16, 32, 64, 128, 256] if not quick else [16, 64]
    noise = [0.0, 0.001, 0.01, 0.05, 0.15] if not quick else [0.0, 0.05]
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [0.0, 1.0]

    save_table(sweep_budget(prob, budgets), TRACK, "sweep_budget")
    save_table(sweep_noise(prob, noise, n_train=128), TRACK, "sweep_noise")

    itr, ite = extrapolation_split(prob)
    frac = len(itr) / n
    ex = study_extrapolation(prob, train_frac=frac)
    save_table(ex, TRACK, "extrapolation")
    save_table(
        sweep_physics_weight(prob, weights, n_train=128), TRACK, "sweep_physics_weight"
    )

    # The complete law on the same split. Deterministic: one fit, no seed.
    ritz = fit_rydberg_ritz(prob.x[itr], prob.y[itr])
    ritz_row = {
        "arm": "ritz",
        "n_free": ritz.n_free,
        **_score_out(prob, ritz, itr, ite),
    }
    save_table(pd.DataFrame([ritz_row]), TRACK, "rydberg_ritz")
    meta["rydberg_ritz"] = ritz_row | {
        "defects": ritz.extra["defects"],
        "L": ritz.params["L"],
    }

    # Prediction 3: the defect structure each arm learned, out of range, per
    # reporting seed, from fits on the extrapolation training split.
    #
    # `pinn_unbalanced` is a diagnostic, not an arm and not a candidate
    # default: the frozen pinn's loss balancing raises the physics weight
    # until the correction is ~1e-5 of the data's spread, so the frozen arm
    # cannot show what a correction would learn. The unbalanced variant
    # (w_phys = 1, fixed) shows it. Nothing is selected on it.
    defect_rows, structure_rows, pinn_rows = [], [], []
    for seed in REPORT_SEEDS if not quick else REPORT_SEEDS[:1]:
        fits = {arm: fit_arm(arm, prob, itr, seed) for arm in ("physics", "pinn", "nn")}
        fits["pinn_unbalanced"] = fit_arm(
            "pinn", prob, itr, seed, pinn_options=pinn_mod.DEFAULT_PINN
        )
        fits["ritz"] = ritz
        for name in ("pinn", "pinn_unbalanced"):
            e = fits[name].extra
            pinn_rows.append(
                {
                    "seed": seed,
                    "variant": name,
                    "w_phys_final": e.get("w_phys_final"),
                    "correction_rms_frac": e.get("correction_rms_frac"),
                    **_score_out(prob, fits[name], itr, ite),
                }
            )
        table = defect_table(prob, fits, ite)
        table.insert(0, "seed", seed)
        defect_rows.append(table)
        for arm in fits:
            structure_rows.append({"seed": seed, **defect_structure(table, arm)})
    save_table(pd.concat(defect_rows, ignore_index=True), TRACK, "defects")
    save_table(pd.DataFrame(structure_rows), TRACK, "defect_structure")
    save_table(pd.DataFrame(pinn_rows), TRACK, "pinn_correction")

    # Headline fit on everything, for the recovered constants.
    allidx = np.arange(n)
    head = {}
    for arm in ("physics", "pinn", "sr"):
        f = fit_arm(arm, prob, allidx, seed=REPORT_SEEDS[0])
        head[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "expression": f.expression,
            "seconds": f.seconds,
        }
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    return meta
