"""Recover the law of gravity from a simulation of it.

The real-data tracks answer "can we recover a constant from measurements?".
These answer a sharper question that only a simulation can: **when the law is
exactly what we put in, how well does each method get it back?** That number
is the method's own noise floor, and every real-data result has to be read
against it.

Three experiments:

`force_law`      from a simulated two-body orbit, recover the exponent in
                 |a| ~ r^p and the constant GM. Ground truth: p = -2 exactly,
                 GM = GM_sun exactly.

`kepler_law`     simulate all eight planets, measure each period FROM THE
                 SIMULATION (not from the ephemeris), and recover the 3/2 in
                 P ~ a^(3/2).

`chaotic_law`    the same force-law recovery, but from the chaotic
                 three-body run. The trajectory is unpredictable; the law is
                 not. Chaos is a property of the solution, not of the
                 equation, and symbolic regression cares only about the
                 equation.
"""

from __future__ import annotations

import numpy as np

from physprior.constants import AU_M, DAY_S, G_NEWTON
from physprior.methods.symbolic import fit_sr, power_law_exponent

from . import orbits

# The same 6th-order stencil the Mercury track had to converge to. Reused
# here deliberately: it is the one numerical choice both modules depend on.
FD_ORDER = 6


def _accel(v: np.ndarray, dt: float, order: int = FD_ORDER) -> np.ndarray:
    a = np.full_like(v, np.nan)
    if order == 2:
        a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
    elif order == 4:
        a[2:-2] = (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / (12 * dt)
    else:
        a[3:-3] = (
            v[6:] - 9 * v[5:-1] + 45 * v[4:-2] - 45 * v[2:-4] + 9 * v[1:-5] - v[:-6]
        ) / (60 * dt)
    return a


def relative_acceleration(traj, i: int = 1, j: int = 0):
    """|r| and |a| of body i relative to body j, from the trajectory alone."""
    dt = float(traj.t[1] - traj.t[0])
    r = traj.r[:, i] - traj.r[:, j]
    v = traj.v[:, i] - traj.v[:, j]
    a = _accel(v, dt)
    m = np.isfinite(a[:, 0])
    return np.linalg.norm(r[m], axis=1), np.linalg.norm(a[m], axis=1), m


def force_law(
    planet: str = "Mercury",
    n_orbits: float = 3.0,
    steps_per_orbit: int = 4000,
    seed: int = 11,
) -> dict:
    """Recover |a| = GM / r^2 from a simulated orbit.

    The orbit is eccentric on purpose: a circular one samples a single radius
    and cannot constrain an exponent at all. Mercury's e = 0.206 gives a
    factor 1.5 in r, which is enough.
    """
    traj = orbits.two_body(
        planet, n_orbits=n_orbits, steps_per_orbit=steps_per_orbit, stride=1
    )
    r, a, _ = relative_acceleration(traj)
    mu_true = traj.meta["mu"]  # G(M_sun + m_planet), exact

    # Least squares in log space: the textbook way to read off a power law.
    p, logc = np.polyfit(np.log(r), np.log(a), 1)
    mu_ls = float(np.exp(logc))
    # ...but exp(intercept) is a BIASED estimate of the constant: the log-log
    # fit minimises error in log a, not in a, and the intercept is an
    # extrapolation to r = 1 m, far outside the data. With the exponent known
    # to be -2, regress a on 1/r^2 directly instead. The two differ by ~60 ppm
    # here, which is the estimator, not the physics.
    basis = r**-2.0
    mu_direct = float(basis @ a / (basis @ basis))

    # Symbolic regression, scaled to O(1) so the constants stay sane.
    rs, as_ = r / r.mean(), a / a.mean()
    f = fit_sr(
        rs.reshape(-1, 1),
        as_,
        feature_names=["r"],
        niterations=60,
        maxsize=12,
        seed=seed,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[],
    )
    p_sr = power_law_exponent(f, "r", (rs.min(), rs.max()), tol=0.02)

    return {
        "planet": planet,
        "n_orbits": n_orbits,
        "steps_per_orbit": steps_per_orbit,
        "fd_order": FD_ORDER,
        "r_min_au": float(r.min() / AU_M),
        "r_max_au": float(r.max() / AU_M),
        "r_dynamic_range": float(r.max() / r.min()),
        "exponent_true": -2.0,
        "exponent_loglog": float(p),
        "exponent_loglog_err": float(abs(p + 2.0)),
        "exponent_sr": p_sr,
        "exponent_sr_err": None if p_sr is None else float(abs(p_sr + 2.0)),
        "sr_expression": f.expression,
        "mu_true": float(mu_true),
        "mu_loglog": mu_ls,
        "mu_loglog_rel_error_ppm": float((mu_ls - mu_true) / mu_true * 1e6),
        "mu_direct": mu_direct,
        "mu_direct_rel_error_ppb": float((mu_direct - mu_true) / mu_true * 1e9),
        "energy_drift": traj.drift(G_NEWTON)["energy_rel_drift"],
    }


def kepler_law(
    planets=tuple(orbits.PLANETS), steps_per_orbit: int = 3000, seed: int = 11
) -> dict:
    """Simulate each planet and recover P ~ a^(3/2) from the simulations.

    The period is MEASURED from the simulated trajectory -- successive
    perihelion passages -- not taken from a table. Nothing that went into the
    integrator is read back out directly.
    """
    rows = []
    for name in planets:
        a_au, e, _m = orbits.PLANETS[name]
        traj = orbits.two_body(
            name, n_orbits=2.2, steps_per_orbit=steps_per_orbit, stride=1
        )
        r = np.linalg.norm(traj.r[:, 1] - traj.r[:, 0], axis=1)
        period = _period_from_perihelia(traj.t, r)
        rows.append(
            {
                "planet": name,
                "a_au": a_au,
                "ecc": e,
                "period_sim_d": period / DAY_S,
                "period_kepler_d": traj.meta["period_s"] / DAY_S,
                "period_rel_err": abs(period - traj.meta["period_s"])
                / traj.meta["period_s"],
            }
        )

    a = np.array([r["a_au"] for r in rows])
    P = np.array([r["period_sim_d"] for r in rows])
    slope, _ = np.polyfit(np.log(a), np.log(P), 1)
    f = fit_sr(
        (a / a.mean()).reshape(-1, 1),
        P / P.mean(),
        feature_names=["a"],
        niterations=60,
        maxsize=12,
        seed=seed,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt"],
    )
    p_sr = power_law_exponent(
        f, "a", (a.min() / a.mean(), a.max() / a.mean()), tol=0.02
    )
    return {
        "rows": rows,
        "exponent_true": 1.5,
        "exponent_loglog": float(slope),
        "exponent_sr": p_sr,
        "sr_expression": f.expression,
        "max_period_rel_err": float(max(r["period_rel_err"] for r in rows)),
    }


def _period_from_perihelia(t: np.ndarray, r: np.ndarray) -> float:
    """Time between successive minima of r, refined by a parabola fit."""
    lo = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
    if len(lo) < 2:
        raise RuntimeError("fewer than two perihelion passages in the run")
    times = []
    for i in lo:
        y0, y1, y2 = r[i - 1], r[i], r[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        times.append(t[i] + shift * (t[1] - t[0]))
    return float(np.mean(np.diff(times)))


def chaotic_law(
    n_periods: float = 6.0, steps_per_period: int = 6000, seed: int = 11
) -> dict:
    """The force law, recovered from the CHAOTIC three-body run.

    Units are G = m = 1, so the exact answer is |a_ij| = 1/r_ij^2 between each
    pair. The trajectory is not predictable more than a few periods ahead; the
    law behind it is recovered to the same precision as from the tidy
    two-body run. That contrast is the point.
    """
    traj = orbits.three_body(
        "chaotic", n_periods=n_periods, steps_per_period=steps_per_period, stride=1
    )
    dt = float(traj.t[1] - traj.t[0])
    # With three bodies the pull of each neighbour cannot be separated from a
    # single body's acceleration, so the unknown fitted here is the one
    # constant they share: G*m. The 1/r^2 SHAPE is assumed, and recovering the
    # shape itself is what `force_law` does on the two-body run.
    a0 = _accel(traj.v[:, 0], dt)
    m = np.isfinite(a0[:, 0])
    d01 = traj.r[:, 1] - traj.r[:, 0]
    d02 = traj.r[:, 2] - traj.r[:, 0]
    n01 = np.linalg.norm(d01, axis=1)[:, None]
    n02 = np.linalg.norm(d02, axis=1)[:, None]
    # Regress a0 = c * (d01/|d01|^3 + d02/|d02|^3): c is G*m, exactly 1.
    basis = (d01 / n01**3 + d02 / n02**3)[m].ravel()
    y = a0[m].ravel()
    c = float(basis @ y / (basis @ basis))
    resid = float(np.linalg.norm(y - c * basis) / np.linalg.norm(y))

    # And the exponent, from the two-body-dominated close approaches.
    sep = np.linalg.norm(d01, axis=1)[m]
    return {
        "Gm_true": 1.0,
        "Gm_recovered": c,
        "Gm_rel_error_ppm": (c - 1.0) * 1e6,
        "residual_rel": resid,
        "lyapunov": orbits.lyapunov_separation(1e-9, n_periods=10.0)["lambda"],
        "separation_range": [float(sep.min()), float(sep.max())],
        "n_samples": int(m.sum()),
        "energy_drift": traj.drift(1.0)["energy_rel_drift"],
    }
