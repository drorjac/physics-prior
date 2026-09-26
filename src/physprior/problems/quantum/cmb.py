"""Track Q -- quantum, thermal. COBE/FIRAS cosmic microwave background.

DATA     The FIRAS monopole spectrum: specific intensity in 43 channels from
         68 to 640 GHz, with 1-sigma uncertainties of order 1e-4 of the peak.
         The most precise blackbody ever measured.

LAW      Planck:  B_nu(T) = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)

PARAM    T, the CMB temperature. And, holding T at the published value, the
         amplitude gives h -- Planck's constant, from the sky.

CAVEAT   The distributed monopole spectrum is built as "a 2.725 K
         blackbody plus the measured residual", so recovering T = 2.725 K is
         partly by construction and is NOT scored as a discovery. What is not
         by construction, and is what this track measures, is: that the SHAPE
         is Planckian to 1e-4 of peak; that symbolic regression finds the
         nu^3/(exp(x)-1) form without being told it; and how each arm behaves
         when it must leave the range it was trained on.

EXTRAPOLATION   Train on the Rayleigh-Jeans side only (nu < 160 GHz, where
         B ~ nu^2) and predict the Wien tail (nu > 160 GHz, where B falls
         exponentially). The training data contains no hint of the turnover.
         This is the cleanest demonstration in the project of what a physics
         prior is actually for.
"""

from __future__ import annotations

import numpy as np
import torch

from physprior.benchmark.protocol import (
    Problem,
    law,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import (
    C_LIGHT,
    H_OVER_K,
    H_PLANCK,
    K_BOLTZMANN,
    T_CMB_K,
    T_CMB_K_ERR,
)
from physprior.data.sources import firas
from physprior.io import save_json, save_table
from physprior.methods.pinn import PhysParam
from physprior.util import first_column

TRACK = "quantum/cmb"
MJY_SR = 1.0e-20  # 1 MJy/sr in W m^-2 sr^-1 Hz^-1
NU_SCALE = 1.0e11  # Hz, for SR
I_SCALE = 1.0e2  # MJy/sr, for SR
TURNOVER_HZ = 1.60e11  # the Planck peak at 2.725 K, 160 GHz


def planck_mjy(nu_hz, T, h=H_PLANCK):
    """Planck function in MJy/sr. `expm1` keeps the Rayleigh-Jeans end exact."""
    nu = np.asarray(nu_hz, float)
    x = h * nu / (K_BOLTZMANN * T)
    return (2.0 * h * nu**3 / C_LIGHT**2) / np.expm1(x) / MJY_SR


@law("2*h*nu**3/c**2 / (exp(h*nu/(k*T)) - 1)")
def law_np(x, T):
    return planck_mjy(first_column(x), T)


def law_t(x, T):
    nu = x[:, 0] if x.ndim > 1 else x
    xx = H_PLANCK * nu / (K_BOLTZMANN * T)
    return (2.0 * H_PLANCK * nu**3 / C_LIGHT**2) / torch.expm1(xx) / MJY_SR


def _sr_transform(x, y):
    X = first_column(x).reshape(-1, 1) / NU_SCALE

    def back(xq, raw_predict):
        return raw_predict(first_column(xq).reshape(-1, 1) / NU_SCALE) * I_SCALE

    return X, np.asarray(y, float) / I_SCALE, back


def problem() -> tuple[Problem, dict]:
    sp = firas.load()
    prob = Problem(
        track=TRACK,
        x=sp.nu_hz.reshape(-1, 1),
        y=sp.intensity,
        law_np=law_np,
        law_t=law_t,
        params=[PhysParam("T", 3.0, positive=True, lo=0.5, hi=20.0)],
        theta_published={"T": T_CMB_K},
        xlabel=r"frequency $\nu$  [GHz]",
        ylabel=r"$I_\nu$  [MJy sr$^{-1}$]",
        sigma=sp.sigma,
        sr_transform=_sr_transform,
        sr_kwargs=dict(
            feature_names=["nu"],
            niterations=80,
            maxsize=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "cube", "square"],
        ),
        nn_cfg=dict(width=32, depth=3, weight_decay=1e-4, epochs=6000),
        notes="COBE/FIRAS monopole, Fixsen et al. 1996 Table 4",
    )
    meta = {
        "provenance": sp.provenance,
        "n_channels": len(sp),
        "nu_ghz_range": [float(sp.nu_hz.min() / 1e9), float(sp.nu_hz.max() / 1e9)],
        "published_T_K": T_CMB_K,
        "published_T_err_K": T_CMB_K_ERR,
        "published_h": H_PLANCK,
        "published_h_over_k": H_OVER_K,
        "turnover_ghz": TURNOVER_HZ / 1e9,
        "caveat": (
            "the distributed monopole = 2.725 K blackbody + measured "
            "residual, so T recovery is partly by construction"
        ),
    }
    return prob, meta


def recover_planck_constant(prob: Problem) -> dict:
    """Hold T at the published value; let the amplitude give h.

    The shape fixes h/kT and the amplitude fixes h, so with T pinned this is
    a genuine measurement of Planck's constant from the microwave sky.
    """
    from scipy.optimize import curve_fit

    nu = prob.x[:, 0]

    def f(nu_, h):
        return planck_mjy(nu_, T_CMB_K, h=h)

    popt, pcov = curve_fit(
        f, nu, prob.y, p0=[6e-34], sigma=prob.sigma, absolute_sigma=True, maxfev=100000
    )
    h_hat, h_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))
    return {
        "h_recovered": h_hat,
        "h_sigma": h_err,
        "h_published": H_PLANCK,
        "rel_error_ppm": (h_hat - H_PLANCK) / H_PLANCK * 1e6,
        "n_sigma": (h_hat - H_PLANCK) / h_err if h_err > 0 else None,
    }


def sr_law_check(
    fit, x_lo: float = 0.02, x_hi: float = 15.0, band=(1.198, 11.26), tol: float = 0.20
) -> dict:
    """Is the discovered expression the Planck law?

    Not judged by a slope against a round number -- an earlier version of this
    check compared the low-frequency log-slope with 2 and called the answer
    wrong, when the exact Planck function's slope over the same interval is
    1.84, because x is not small enough there for Rayleigh-Jeans to hold.

    The question is asked directly instead: evaluate the expression and the
    exact Planck function on the same grid and compare, separately INSIDE the
    fitted band and OUTSIDE it. Any decent fit matches inside. Only the right
    law matches outside.
    """
    import sympy

    out = {
        "expression": fit.expression,
        "max_rel_dev_in_band": None,
        "max_rel_dev_below": None,
        "max_rel_dev_above": None,
        "is_planck_form": False,
    }
    try:
        nu_s = sympy.Symbol("nu")
        f = sympy.lambdify(nu_s, sympy.sympify(fit.expression), "numpy")
        x = np.geomspace(x_lo, x_hi, 400)
        nu = x * K_BOLTZMANN * T_CMB_K / H_PLANCK
        truth = planck_mjy(nu, T_CMB_K)
        got = np.asarray(f(nu / NU_SCALE), float) * I_SCALE
        if not np.all(np.isfinite(got)):
            return out
        rel = np.abs(got - truth) / truth
        inb = (x >= band[0]) & (x <= band[1])
        out["max_rel_dev_in_band"] = float(rel[inb].max()) if inb.any() else None
        out["max_rel_dev_below"] = float(rel[x < band[0]].max())
        out["max_rel_dev_above"] = float(rel[x > band[1]].max())
        out["max_rel_dev_outside"] = float(rel[~inb].max())
        out["is_planck_form"] = bool(out["max_rel_dev_outside"] < tol)
    except Exception:
        pass
    return out


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    print(
        f"[{TRACK}] {len(prob)} channels, "
        f"{meta['nu_ghz_range'][0]:.0f}-{meta['nu_ghz_range'][1]:.0f} GHz"
    )

    budgets = [5, 8, 12, 20, 30] if not quick else [8, 20]
    noise = [0.0, 0.001, 0.01, 0.05, 0.15] if not quick else [0.0, 0.05]
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [0.0, 1.0]

    save_table(sweep_budget(prob, budgets), TRACK, "sweep_budget")
    save_table(sweep_noise(prob, noise, n_train=30), TRACK, "sweep_noise")
    # Train on the Rayleigh-Jeans side only: nu < 160 GHz is 21 of 43 channels.
    frac = float(np.mean(prob.x[:, 0] < TURNOVER_HZ))
    save_table(study_extrapolation(prob, train_frac=frac), TRACK, "extrapolation")
    save_table(
        sweep_physics_weight(prob, weights, n_train=30), TRACK, "sweep_physics_weight"
    )
    meta["extrapolation_train_frac"] = frac

    from physprior.benchmark.protocol import fit_arm

    allidx = np.arange(len(prob))
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
    head["planck_constant"] = recover_planck_constant(prob)
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    return meta


def band_coverage_control(
    x_mins=(1.2, 0.6, 0.3, 0.1, 0.03),
    n_channels=43,
    noise_frac=7.4e-4,
    niterations=120,
    maxsize=22,
    seed=11,
) -> list[dict]:
    """Why did SR miss Planck's law -- the method, or the data?

    FIRAS covers x = h nu / k T from 1.2 to 11.3: almost all Wien, almost no
    Rayleigh-Jeans. In that band exp(-x) and 1/(exp(x)-1) are nearly the same
    function, so the denominator is simply not identifiable.

    The control keeps the method, the channel count and the noise fixed and
    only widens the band downwards, on a SYNTHETIC Planck spectrum where the
    right answer is known. If SR starts finding the Planck form as x_min
    falls, the failure on FIRAS is a property of the data, not of SR.
    """
    from physprior.methods.symbolic import fit_sr

    rows = []
    for xmin in x_mins:
        nu = np.geomspace(xmin, 11.26, n_channels) * K_BOLTZMANN * T_CMB_K / H_PLANCK
        y = planck_mjy(nu, T_CMB_K)
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise_frac * y.max(), len(y))
        X = (nu / NU_SCALE).reshape(-1, 1)
        f = fit_sr(
            X,
            y / I_SCALE,
            feature_names=["nu"],
            niterations=niterations,
            maxsize=maxsize,
            seed=seed,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "cube", "square"],
        )
        chk = sr_law_check(f)
        rows.append(
            {
                "x_min": xmin,
                "x_max": 11.26,
                "nu_min_ghz": float(nu.min() / 1e9),
                "decades_below_peak": float(np.log10(2.821 / xmin)),
                **chk,
            }
        )
        print(
            f"  x_min={xmin:5.2f}  planck={chk['is_planck_form']}  "
            f"dev_outside={chk.get('max_rel_dev_outside')}",
            flush=True,
        )
    return rows
