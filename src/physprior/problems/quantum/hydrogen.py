"""Track A -- quantum, atomic. NIST hydrogen levels.

DATA     E_n, the energy of the n-th level of H I above the ground state, in
         cm^-1, for n = 1..40, with quoted uncertainties. NIST evaluated
         reference values, anchored to measurement (see data/sources/nist.py).

LAW      Bohr:   E_n = R (1 - 1/n^2)

PARAM    R, the Rydberg constant for hydrogen.

WHY THIS TRACK   It is the cleanest law-discovery problem in physics that has
         real data behind it, and it has a twist worth more than the fit: the
         law is not exact. Fitting R to the data returns the measured
         ionisation limit, 109678.77 cm^-1, while Bohr-with-reduced-mass
         predicts 109677.58. The 10.8 ppm gap is the relativistic and QED
         (Lamb shift) correction. A method that reports a number with an
         error bar can SEE that gap. A method that reports only a curve
         cannot, however well it fits.

EXTRAPOLATION    train on n <= 10, predict n = 11..40. This is the test the
         black box cannot pass: nothing in the training range tells it the
         levels converge to a finite limit.
"""

from __future__ import annotations

import numpy as np

from physprior.benchmark.protocol import (
    Problem,
    law,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import RYDBERG_H_CM
from physprior.data.sources import nist as hydrogen
from physprior.io import save_json, save_table
from physprior.methods.pinn import PhysParam
from physprior.util import first_column

TRACK = "quantum/hydrogen"

# The `pinn` arm's physics weight, chosen on the tuning seeds by
# `physprior.benchmark.pinn_tuning` (results/quantum/hydrogen/tune/w_phys_selection.csv).
PINN_W_PHYS = 1
SR_SCALE = 1.0e5  # cm^-1 -> O(1), so PySR's constants stay sane


@law("R*(1 - 1/n**2)")
def law_np(x, R):
    n = first_column(x)
    return R * (1.0 - 1.0 / n**2)


def law_t(x, R):
    n = x[:, 0] if x.ndim > 1 else x
    return R * (1.0 - 1.0 / n**2)


def _sr_transform(x, y):
    """SR sees n and E/1e5; the back-transform undoes the scaling."""
    X = first_column(x).reshape(-1, 1)

    def back(xq, raw_predict):
        return raw_predict(first_column(xq).reshape(-1, 1)) * SR_SCALE

    return X, np.asarray(y, float) / SR_SCALE, back


def problem() -> tuple[Problem, dict]:
    lv = hydrogen.load()
    prob = Problem(
        track=TRACK,
        pinn_w_phys=PINN_W_PHYS,
        x=lv.n.reshape(-1, 1),
        y=lv.energy_icm,
        law_np=law_np,
        law_t=law_t,
        params=[PhysParam("R", 1.0e5, positive=True, lo=1.0e4, hi=1.0e6)],
        theta_published={"R": RYDBERG_H_CM},
        xlabel="principal quantum number n",
        ylabel="E_n  [cm$^{-1}$]",
        sigma=np.where(
            np.isfinite(lv.sigma_icm) & (lv.sigma_icm > 0), lv.sigma_icm, 1e-3
        ),
        sr_transform=_sr_transform,
        sr_kwargs=dict(
            feature_names=["n"],
            niterations=60,
            maxsize=16,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "inv"],
        ),
        nn_cfg=dict(width=32, depth=3, weight_decay=1e-4, epochs=6000),
        notes="NIST ASD H I n-averaged levels, n=1..40",
    )
    meta = {
        "provenance": lv.provenance,
        "n_levels": len(lv),
        "ionisation_limit_icm": lv.limit_icm,
        "ionisation_limit_sigma_icm": lv.limit_sigma_icm,
        "bohr_rydberg_H_icm": RYDBERG_H_CM,
        "limit_minus_bohr_icm": lv.limit_icm - RYDBERG_H_CM,
        "limit_minus_bohr_ppm": (lv.limit_icm - RYDBERG_H_CM) / RYDBERG_H_CM * 1e6,
    }
    return prob, meta


def sr_law_check(fit, n_range=(2.0, 40.0)) -> dict:
    """Read the physics off a discovered expression.

    If the expression really is `L - c/n^p`, then L is the ionisation limit,
    c the Rydberg constant and p must be 2. All three are measured from the
    function itself, not parsed out of its text.
    """
    import sympy

    out = {
        "expression": fit.expression,
        "limit": None,
        "coeff": None,
        "power": None,
        "is_bohr_form": False,
    }
    try:
        n = sympy.Symbol("n")
        f = sympy.lambdify(n, sympy.sympify(fit.expression), "numpy")
        big = np.array([1e6, 1e7, 1e8])
        L = float(np.mean(np.asarray(f(big), float))) * SR_SCALE
        ns = np.geomspace(float(n_range[0]), float(n_range[1]), 48)
        d = L - np.asarray(f(ns), float) * SR_SCALE  # should be c/n^p
        good = d > 0
        if good.sum() < 8:
            return out
        p = np.polyfit(np.log(ns[good]), np.log(d[good]), 1)
        out["limit"] = L
        out["power"] = float(-p[0])
        out["coeff"] = float(np.exp(p[1]))
        resid = np.std(np.log(d[good]) - np.polyval(p, np.log(ns[good])))
        out["loglog_resid"] = float(resid)
        out["is_bohr_form"] = bool(abs(out["power"] - 2.0) < 0.05 and resid < 0.02)
    except Exception:
        pass
    return out


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    n = len(prob)
    print(
        f"[{TRACK}] {n} levels, n = {prob.x[:, 0].min():.0f}..{prob.x[:, 0].max():.0f}"
    )

    budgets = [4, 6, 8, 12, 20, 30] if not quick else [6, 12]
    noise = [0.0, 0.001, 0.01, 0.05, 0.15] if not quick else [0.0, 0.05]
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [0.0, 1.0]

    b = sweep_budget(prob, budgets)
    save_table(b, TRACK, "sweep_budget")
    nz = sweep_noise(prob, noise, n_train=25)
    save_table(nz, TRACK, "sweep_noise")
    ex = study_extrapolation(prob, train_frac=0.25)  # n <= 10 -> n = 11..40
    save_table(ex, TRACK, "extrapolation")
    wp = sweep_physics_weight(prob, weights, n_train=25)
    save_table(wp, TRACK, "sweep_physics_weight")

    # Headline fit on everything, for the recovered constant and the law.
    from physprior.benchmark.protocol import fit_arm

    allidx = np.arange(n)
    head = {}
    for arm in ("physics", "pinn", "sr"):
        f = fit_arm(arm, prob, allidx, seed=11)
        head[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "expression": f.expression,
            "seconds": f.seconds,
        }
        if arm == "sr":
            head[arm]["law_check"] = sr_law_check(f)
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    return meta
