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

WHAT IS LEFT, AND WHAT IT WAS. Once the derivative has converged, alpha
still sits about 1e-4 above 1, at 7 formal sigma. `neglected_terms` puts
back, one at a time, what DE441 has and this model does not:
  - the Sun's oblateness (J2) and frame dragging -- alpha moves AWAY from 1;
  - the n-body relativistic (EIH) equations in the barycentric frame, in
    place of the one-body Schwarzschild term -- alpha overshoots the other
    way;
  - both together, i.e. DE441's own model -- alpha = 1 within its error.
Two omissions of opposite sign, each ~4e-4, had nearly cancelled. A model
can fit to 1e-10 while its one free coefficient absorbs both.

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
    J2_SUN,
    MERCURY_GR_PRECESSION_ARCSEC_CY,
    OBLIQUITY_J2000_DEG,
    R_SUN_DE440_M,
    SUN_C_OVER_MR2,
    SUN_POLE_DEC_DEG,
    SUN_POLE_RA_DEG,
    SUN_ROTATION_DEG_PER_DAY,
)
from physprior.data.sources import horizons as eph
from physprior.io import save_json, save_table
from physprior.units import require

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
        v=v,
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


def _sun_pole_ecliptic() -> np.ndarray:
    """The Sun's spin axis in the ecliptic J2000 frame the vectors come in."""
    ra, dec, eps = np.radians([SUN_POLE_RA_DEG, SUN_POLE_DEC_DEG, OBLIQUITY_J2000_DEG])
    x, y, z = np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)
    return np.array(
        [x, np.cos(eps) * y + np.sin(eps) * z, -np.sin(eps) * y + np.cos(eps) * z]
    )


def _solar_figure_terms(r: np.ndarray, v: np.ndarray, rn: np.ndarray):
    """Accelerations from the Sun's oblateness (J2) and its spin (Lense-Thirring).

    Both are in DE441 and neither is in `gr_regression`. Values are the
    ephemeris' own (`constants.py`, [DE440]); nothing here is fitted.
    """
    k = _sun_pole_ecliptic()
    z = (r @ k)[:, None]
    a_j2_per_unit = (1.5 * GM_SUN * R_SUN_DE440_M**2 / rn**5) * (
        (5 * z**2 / rn**2 - 1) * r - 2 * z * k
    )
    # G * S_sun, with S_sun = C M R^2 omega -- so G M enters, never G alone.
    omega = np.radians(SUN_ROTATION_DEG_PER_DAY) / DAY_S
    gs = SUN_C_OVER_MR2 * GM_SUN * R_SUN_DE440_M**2 * omega * k
    a_lt = (2 / (C_LIGHT**2 * rn**3)) * (
        3 * (r @ gs)[:, None] * np.cross(r, v) / rn**2 + np.cross(v, gs)
    )
    return a_j2_per_unit, a_lt


def _lstsq(y: np.ndarray, cols: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    A = np.stack(cols, axis=1)
    cs = np.linalg.norm(A, axis=0)
    coef, *_ = np.linalg.lstsq(A / cs, y, rcond=None)
    coef = coef / cs
    res = y - A @ coef
    s2 = res @ res / (len(y) - A.shape[1])
    cov = s2 * np.linalg.inv((A / cs).T @ (A / cs)) / np.outer(cs, cs)
    return coef, np.sqrt(np.diag(cov))


def _eih_1pn(R: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The 1PN part of the Einstein-Infeld-Hoffmann acceleration, beta = gamma = 1.

    R, V: (B, N, 3) barycentric positions and velocities; mu: (B,) GMs.
    This is the point-mass relativistic part of the equations DE441
    integrates (Park et al. 2021, eq. 1 with beta = gamma = 1). The Newtonian
    accelerations inside it are the point-mass ones, which is exact at 1PN.
    """
    n_bodies = len(mu)
    c2 = C_LIGHT**2

    def dot(a, b):
        return np.sum(a * b, axis=-1)[..., None]

    a_n = np.zeros_like(R)
    pot = np.zeros((*R.shape[:2], 1))  # sum over k of mu_k / r_ik
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                d = R[j] - R[i]
                rij = np.linalg.norm(d, axis=1)[:, None]
                a_n[i] += mu[j] * d / rij**3
                pot[i] += mu[j] / rij
    a1 = np.zeros_like(R)
    for i in range(n_bodies):
        vi2 = dot(V[i], V[i])
        for j in range(n_bodies):
            if i == j:
                continue
            rji = R[j] - R[i]
            rij = np.linalg.norm(rji, axis=1)[:, None]
            nij = -rji / rij
            bracket = (
                -4 * pot[i]
                - pot[j]
                + vi2
                + 2 * dot(V[j], V[j])
                - 4 * dot(V[i], V[j])
                - 1.5 * dot(nij, V[j]) ** 2
                + 0.5 * dot(rji, a_n[j])
            )
            a1[i] += mu[j] * rji / rij**3 * bracket / c2
            a1[i] += (
                mu[j] / rij**3 * dot(-rji, 4 * V[i] - 3 * V[j]) * (V[i] - V[j]) / c2
            )
            a1[i] += 3.5 * mu[j] * a_n[j] / rij / c2
    return a1


def _eih_heliocentric(d: dict, step: str) -> np.ndarray:
    """Mercury's heliocentric relativistic acceleration from the n-body EIH terms.

    Heliocentric acceleration is barycentric Mercury minus barycentric Sun,
    so the relativistic part is the difference of their 1PN terms. Every
    barycentric state is (heliocentric state + the Sun's barycentric state),
    so the Sun is the only body fetched about the barycentre.
    """
    sun = eph.vectors("10", SPAN[0], SPAN[1], step, centre="ssb")
    require(bool(np.allclose(sun.jd, d["jd"])), "Mercury: barycentre grid mismatch")
    zero = np.zeros_like(d["r"])
    r_h, v_h = [zero, d["r"]], [zero, d["v"]]
    for cmd in PERTURBERS:
        other = eph.vectors(cmd, SPAN[0], SPAN[1], step)
        r_h.append(other.r_m)
        v_h.append(other.v_ms)
    R = np.array(r_h) + sun.r_m[None]
    V = np.array(v_h) + sun.v_ms[None]
    mu = np.array([GM_SUN, GM_MERCURY] + [gm * 1e9 for gm in PERTURBERS.values()])
    a1 = _eih_1pn(R, V, mu)
    return a1[1] - a1[0]


def neglected_terms(configs=(("180m", 6), ("90m", 6))) -> list[dict]:
    """Does putting back what the model leaves out bring alpha to 1?

    Six models of the same acceleration, per converged derivative: as
    shipped; minus DE441's solar J2; minus J2 and Lense-Thirring; J2 left
    free, so the data say how much oblateness they want; the barycentric
    n-body relativistic term in place of the one-body one; and that term with
    J2 and Lense-Thirring removed -- DE441's own model.
    """
    rows = []
    for step, order in configs:
        d = _assemble(step, order)
        m, r, rn = d["mask"], d["r"], d["rn"]
        a_j2, a_lt = _solar_figure_terms(r, d["v"], rn)
        a_eih = _eih_heliocentric(d, step)
        base = d["a_obs"] - d["a_pert"]
        figure = base - J2_SUN * a_j2 - a_lt
        newton = (-r / rn**3)[m].ravel()
        gr, eih = d["a_gr"][m].ravel(), a_eih[m].ravel()
        for model, y, rel, extra in (
            ("as_shipped", base, gr, []),
            ("minus_J2", base - J2_SUN * a_j2, gr, []),
            ("minus_J2_and_LT", figure, gr, []),
            ("J2_free", base, gr, [a_j2[m].ravel()]),
            ("EIH", base, eih, []),
            ("EIH_minus_J2_and_LT", figure, eih, []),
        ):
            coef, err = _lstsq(y[m].ravel(), [newton, rel, *extra])
            rows.append(
                {
                    "step": step,
                    "fd_order": order,
                    "model": model,
                    "alpha_GR": float(coef[1]),
                    "alpha_sigma": float(err[1]),
                    "alpha_minus_one_sigmas": float((coef[1] - 1.0) / err[1]),
                    "J2_fitted": float(coef[2]) if extra else float("nan"),
                    "J2_fitted_sigma": float(err[2]) if extra else float("nan"),
                }
            )
    return rows


def richardson_truncation(rows: list[dict], order: int = FD_ORDER) -> dict:
    """How much of alpha is still the derivative, from two steps a factor 2 apart.

    If the stencil's error goes as h^p, halving h divides it by 2^p, so the
    finer value's remaining error is (difference) / (2^p - 1).
    """
    at = {r["step"]: r["alpha_GR"] for r in rows if r.get("fd_order") == order}
    coarse, fine = at["180m"], at["90m"]
    diff = coarse - fine
    return {
        "order": order,
        "alpha_coarse": coarse,
        "alpha_fine": fine,
        "step_change": diff,
        "fine_truncation_estimate": diff / (2**order - 1),
        "alpha_extrapolated": fine - diff / (2**order - 1),
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
    meta["truncation"] = richardson_truncation(meta["gr_convergence"])
    meta["neglected_terms"] = neglected_terms()
    save_table(pd.DataFrame(meta["neglected_terms"]), TRACK, "neglected_terms")
    save_json(meta, TRACK, "meta")
    return meta
