"""The relativistic correction, measured in Mercury's acceleration.

DATA   Mercury's heliocentric position and velocity from JPL DE441, plus every
       other planet's position on the same grid.

MODEL  Mercury's acceleration, obtained by differentiating the tabulated
       velocity, is written as

           a = -G(M_sun + m) r/r^3
               + SUM_j G m_j [ (r_j - r)/|r_j - r|^3 - r_j/|r_j|^3 ]
               + alpha * a_GR,

       with a_GR the 1PN Schwarzschild term. Every perturber position is real
       data, so the only unknowns are GM_sun and `alpha`, and the model is
       LINEAR in both: one least-squares solve, no integration.

       General relativity says alpha = 1.

A WARNING THIS TRACK EARNED. The GR term is 8e-8 of the main term. At a
3-hour step with a 4th-order derivative the truncation error is itself a few
times 1e-9 and alpha comes out 1.13 with a formal error of 0.002 -- a 13%
"violation of general relativity" at 56 sigma, entirely numerical. The
convergence study is not an appendix, it is the result.

The companion simulation is `physprior.problems.relativity.spacetime`, which integrates the
Schwarzschild orbit equation directly and gets the same 43 arcsec/century from
the other direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from physprior.constants import (
    C_LIGHT,
    DAY_S,
    GM_SUN,
    MERCURY_GR_PRECESSION_ARCSEC_CY,
)
from physprior.data.sources import horizons as eph
from physprior.io import save_json, save_table

TRACK = "relativity/mercury"

# DE441 system GMs in km^3/s^2, by Horizons barycentre id.
PERTURBERS = {
    "2": 3.24858592000e5,  # Venus
    "3": 4.03503235625e5,  # Earth-Moon barycentre
    "4": 4.28283758160e4,  # Mars system
    "5": 1.26712764100e8,  # Jupiter system
    "6": 3.79405848418e7,  # Saturn system
    "7": 5.79455640000e6,  # Uranus system
    "8": 6.83652710058e6,  # Neptune system
}
GM_MERCURY = 2.2031868551e4 * 1e9  # m^3/s^2

# Chosen by the convergence study below, not by taste.
FD_ORDER = 6
FD_STEP = "180m"
SPAN = ("2020-01-01", "2020-07-20")  # ~2 Mercury orbits


def _finite_difference(v: np.ndarray, h: float, order: int) -> np.ndarray:
    a = np.full_like(v, np.nan)
    if order == 2:
        a[1:-1] = (v[2:] - v[:-2]) / (2 * h)
    elif order == 4:
        a[2:-2] = (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / (12 * h)
    elif order == 6:
        a[3:-3] = (
            v[6:] - 9 * v[5:-1] + 45 * v[4:-2] - 45 * v[2:-4] + 9 * v[1:-5] - v[:-6]
        ) / (60 * h)
    else:
        raise ValueError(f"unsupported stencil order {order}")
    return a


def _assemble(step: str = FD_STEP, order: int = FD_ORDER, span=SPAN):
    """Build (acceleration, Newtonian columns, GR column) from real data."""
    me = eph.vectors("Mercury", span[0], span[1], step)
    h = (me.jd[1] - me.jd[0]) * DAY_S
    a_obs = _finite_difference(me.v_ms, h, order)
    r, v = me.r_m, me.v_ms
    rn = np.linalg.norm(r, axis=1)[:, None]

    a_pert = np.zeros_like(r)
    for cmd, gm_km in PERTURBERS.items():
        other = eph.vectors(cmd, span[0], span[1], step)
        assert np.allclose(other.jd, me.jd), "perturber grid mismatch"
        gm = gm_km * 1e9
        d = other.r_m - r
        a_pert += gm * (
            d / np.linalg.norm(d, axis=1)[:, None] ** 3
            - other.r_m / np.linalg.norm(other.r_m, axis=1)[:, None] ** 3
        )

    vn2 = np.sum(v * v, axis=1)[:, None]
    rv = np.sum(r * v, axis=1)[:, None]
    a_gr = (GM_SUN / (C_LIGHT**2 * rn**3)) * ((4 * GM_SUN / rn - vn2) * r + 4 * rv * v)
    m = np.isfinite(a_obs[:, 0])
    return dict(
        a_obs=a_obs,
        a_pert=a_pert,
        a_gr=a_gr,
        r=r,
        rn=rn,
        mask=m,
        jd=me.jd,
        scale=float(np.linalg.norm(a_obs[m], axis=1).mean()),
        provenance=me.provenance,
    )


def gr_regression(step: str = FD_STEP, order: int = FD_ORDER, span=SPAN) -> dict:
    d = _assemble(step, order, span)
    m, r, rn = d["mask"], d["r"], d["rn"]
    y = (d["a_obs"] - d["a_pert"])[m].ravel()
    A = np.stack([(-r / rn**3)[m].ravel(), d["a_gr"][m].ravel()], axis=1)
    cs = np.linalg.norm(A, axis=0)
    coef, *_ = np.linalg.lstsq(A / cs, y, rcond=None)
    coef = coef / cs
    res = y - A @ coef
    s2 = res @ res / (len(y) - 2)
    cov = s2 * np.linalg.inv((A / cs).T @ (A / cs)) / np.outer(cs, cs)
    err = np.sqrt(np.diag(cov))

    gm_truth = GM_SUN + GM_MERCURY
    a_sun = -gm_truth * r / rn**3
    ladder = {
        "sun_only": float(
            np.linalg.norm((d["a_obs"] - a_sun)[m], axis=1).mean() / d["scale"]
        ),
        "sun_planets": float(
            np.linalg.norm((d["a_obs"] - a_sun - d["a_pert"])[m], axis=1).mean()
            / d["scale"]
        ),
        "sun_planets_gr": float(
            np.linalg.norm(
                (d["a_obs"] - a_sun - d["a_pert"] - d["a_gr"])[m], axis=1
            ).mean()
            / d["scale"]
        ),
    }
    return {
        "step": step,
        "fd_order": order,
        "n_epochs": int(d["mask"].sum()),
        "span": list(span),
        "GM_sun_plus_mercury": float(coef[0]),
        "GM_sigma": float(err[0]),
        "GM_truth": gm_truth,
        "GM_rel_error_ppb": float((coef[0] - gm_truth) / gm_truth * 1e9),
        "alpha_GR": float(coef[1]),
        "alpha_sigma": float(err[1]),
        "alpha_minus_one": float(coef[1] - 1.0),
        "residual_ladder": ladder,
        "gr_signal_fraction": ladder["sun_planets"],
        "precession_arcsec_cy": float(coef[1] * MERCURY_GR_PRECESSION_ARCSEC_CY),
        "precession_published": MERCURY_GR_PRECESSION_ARCSEC_CY,
        "provenance": d["provenance"],
    }


def derivative_convergence(
    configs=(
        ("360m", 4),
        ("180m", 2),
        ("180m", 4),
        ("180m", 6),
        ("90m", 4),
        ("90m", 6),
    ),
) -> list[dict]:
    """The result is not alpha, it is alpha once it has stopped moving."""
    rows = []
    for step, order in configs:
        try:
            g = gr_regression(step, order)
            rows.append(
                {
                    k: g[k]
                    for k in (
                        "step",
                        "fd_order",
                        "n_epochs",
                        "alpha_GR",
                        "alpha_sigma",
                        "GM_rel_error_ppb",
                    )
                }
                | {"residual_after_gr": g["residual_ladder"]["sun_planets_gr"]}
            )
        except Exception as e:
            rows.append({"step": step, "fd_order": order, "error": str(e)[:150]})
        print(
            f"  step={step:>5s} order={order}: "
            f"alpha={rows[-1].get('alpha_GR', float('nan')):.5f}",
            flush=True,
        )
    return rows


def run(quick: bool = False) -> dict:
    import pandas as pd

    print(f"[{TRACK}] relativistic correction from Mercury's acceleration")
    meta: dict[str, Any] = {"gr": gr_regression()}
    meta["gr_convergence"] = derivative_convergence()
    save_table(pd.DataFrame(meta["gr_convergence"]), TRACK, "gr_convergence")
    save_json(meta, TRACK, "meta")
    return meta
