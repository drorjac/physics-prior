"""Gravity simulations. The ones from a first- or second-year mechanics course.

`two_body`        Sun + one planet. The Kepler problem, with the ellipse, the
                  conserved energy and angular momentum, and the equal-areas
                  law all measurable from the output.

`three_body`      Two named cases. The Euler/Lagrange figure-eight
                  choreography (Chenciner & Montgomery 2000), which is a
                  stable periodic solution of the three-body problem; and a
                  perturbed copy of it, which is not -- two runs whose initial
                  conditions differ by one part in 10^9 separate completely.
                  That is Lyapunov divergence, measured rather than asserted.

`solar_system`    The eight planets from real JPL initial conditions,
                  integrated forward and compared against the real ephemeris,
                  so the simulation's error is a measured quantity.

Every run is a `Trajectory` and carries its own conservation diagnostics, so
"is this integration trustworthy?" is answered in the same object that answers
"what happened?".
"""

from __future__ import annotations

import numpy as np

from physprior.constants import AU_M, DAY_S, G_NEWTON, GM_SUN
from physprior.numerics.integrators import Trajectory, integrate

M_SUN_KG = GM_SUN / G_NEWTON
YEAR_S = 365.25 * DAY_S

# Planet data for the two-body demos: a [AU], e, mass [kg].
PLANETS = {
    "Mercury": (0.38710, 0.20563, 3.3011e23),
    "Venus": (0.72333, 0.00677, 4.8675e24),
    "Earth": (1.00000, 0.01671, 5.97237e24),
    "Mars": (1.52371, 0.09339, 6.4171e23),
    "Jupiter": (5.20288, 0.04839, 1.89819e27),
    "Saturn": (9.53667, 0.05386, 5.6834e26),
    "Uranus": (19.18916, 0.04726, 8.6813e25),
    "Neptune": (30.06992, 0.00859, 1.02413e26),
}


def kepler_period(a_m: float, m_total_kg: float) -> float:
    return 2.0 * np.pi * np.sqrt(a_m**3 / (G_NEWTON * m_total_kg))


def two_body(
    planet: str = "Mercury",
    n_orbits: float = 3.0,
    steps_per_orbit: int = 2000,
    method: str = "verlet",
    stride: int = 4,
) -> Trajectory:
    """Sun + one planet, started at perihelion in the orbital plane.

    The vis-viva equation fixes the perihelion speed from `a` and `e`:
        v_p = sqrt( G M (1+e) / (a (1-e)) )
    so the orbit that comes out has the intended shape, and any departure is
    the integrator's, not the initial condition's.
    """
    a_au, e, m_p = PLANETS[planet]
    a = a_au * AU_M
    mu = G_NEWTON * (M_SUN_KG + m_p)
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)
    dt = period / steps_per_orbit

    # Start in the centre-of-mass frame so the pair does not drift off screen.
    m_tot = M_SUN_KG + m_p
    r0 = np.array([[-m_p / m_tot * r_p, 0.0, 0.0], [M_SUN_KG / m_tot * r_p, 0.0, 0.0]])
    v0 = np.array([[0.0, -m_p / m_tot * v_p, 0.0], [0.0, M_SUN_KG / m_tot * v_p, 0.0]])

    return integrate(
        [M_SUN_KG, m_p],
        r0,
        v0,
        dt=dt,
        n_steps=int(n_orbits * steps_per_orbit),
        G=G_NEWTON,
        method=method,
        stride=stride,
        names=["Sun", planet],
        meta={
            "planet": planet,
            "a_m": a,
            "ecc": e,
            "period_s": period,
            "mu": mu,
            "steps_per_orbit": steps_per_orbit,
        },
    )


# Chenciner & Montgomery's figure-eight, in the units where G = m = 1.
FIGURE8_R = np.array(
    [[0.97000436, -0.24308753, 0.0], [-0.97000436, 0.24308753, 0.0], [0.0, 0.0, 0.0]]
)
FIGURE8_V = np.array(
    [
        [0.466203685, 0.43236573, 0.0],
        [0.466203685, 0.43236573, 0.0],
        [-0.93240737, -0.86473146, 0.0],
    ]
)
FIGURE8_PERIOD = 6.32591398


def three_body(
    case: str = "figure8",
    n_periods: float = 3.0,
    steps_per_period: int = 4000,
    perturbation: float = 0.0,
    method: str = "verlet",
    stride: int = 8,
) -> Trajectory:
    """The three-body problem in natural units (G = 1, each mass = 1).

    `case="figure8"`   the periodic choreography: all three bodies chase each
                       other around the same figure-eight curve.
    `case="chaotic"`   the same, with `perturbation` added to one coordinate.
                       Nothing else changes.
    """
    r0 = FIGURE8_R.copy()
    v0 = FIGURE8_V.copy()
    if case == "chaotic" or perturbation:
        r0[0, 0] += perturbation or 1e-9
    dt = FIGURE8_PERIOD / steps_per_period
    return integrate(
        [1.0, 1.0, 1.0],
        r0,
        v0,
        dt=dt,
        n_steps=int(n_periods * steps_per_period),
        G=1.0,
        method=method,
        stride=stride,
        names=["body 1", "body 2", "body 3"],
        meta={
            "case": case,
            "perturbation": perturbation,
            "period": FIGURE8_PERIOD,
            "units": "G = m = 1",
        },
    )


def lyapunov_separation(
    perturbation: float = 1e-9, n_periods: float = 12.0, steps_per_period: int = 4000
) -> dict:
    """Run the figure-eight twice, differing by `perturbation`, and measure
    how fast the two copies separate. Chaos is a measurement, not an adjective.
    """
    a = three_body("figure8", n_periods, steps_per_period, stride=8)
    b = three_body(
        "figure8", n_periods, steps_per_period, perturbation=perturbation, stride=8
    )
    sep = np.linalg.norm(a.r - b.r, axis=-1).max(axis=1)
    sep = np.maximum(sep, 1e-18)
    t = a.t
    # Fit the exponential growth over the window before saturation.
    scale = np.linalg.norm(a.r, axis=-1).max()
    grow = sep < 0.1 * scale
    lam = np.nan
    if grow.sum() > 20:
        lam = float(np.polyfit(t[grow], np.log(sep[grow]), 1)[0])
    return {
        "t": t,
        "separation": sep,
        "lambda": lam,
        "perturbation": perturbation,
        "t_period": FIGURE8_PERIOD,
        "saturation_scale": float(scale),
        "e_folding_periods": (FIGURE8_PERIOD * lam)
        and float(1.0 / (lam * FIGURE8_PERIOD))
        if np.isfinite(lam) and lam > 0
        else None,
    }


def solar_system(
    years: float = 10.0,
    dt_days: float = 1.0,
    stride: int = 20,
    method: str = "verlet",
    epoch: str = "2020-01-01",
) -> Trajectory:
    """The eight planets from REAL JPL initial conditions.

    Only the state at `epoch` is taken from the ephemeris; everything after is
    this integrator's own Newtonian solution. Comparing the two later is a
    genuine test, not a fit.
    """
    import datetime as _dt

    from physprior.data.sources import horizons as eph

    names, masses, r0, v0 = ["Sun"], [M_SUN_KG], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]
    # Only the FIRST row is used -- the state at the epoch. A 30-day span is
    # requested rather than a 2-day one because the loader's guard against a
    # silently-truncated Horizons reply requires more than ten rows, and that
    # guard is worth keeping strict.
    stop = (_dt.date.fromisoformat(epoch) + _dt.timedelta(days=30)).isoformat()
    for name, (_cmd, gm_km) in eph.BODIES.items():
        v = eph.vectors(name, epoch, stop, "1d")
        names.append(name)
        masses.append(gm_km * 1e9 / G_NEWTON)
        r0.append(v.r_m[0])
        v0.append(v.v_ms[0])

    m_arr = np.asarray(masses)
    r_arr, v_arr = np.asarray(r0), np.asarray(v0)
    # Work in the barycentric frame: the Sun's recoil is part of the physics.
    v_arr -= np.sum(m_arr[:, None] * v_arr, axis=0) / m_arr.sum()
    r_arr -= np.sum(m_arr[:, None] * r_arr, axis=0) / m_arr.sum()

    dt = dt_days * DAY_S
    return integrate(
        m_arr,
        r_arr,
        v_arr,
        dt=dt,
        n_steps=int(years * YEAR_S / dt),
        G=G_NEWTON,
        method=method,
        stride=stride,
        names=names,
        meta={
            "epoch": epoch,
            "years": years,
            "dt_days": dt_days,
            "source": "JPL Horizons DE441 initial state",
        },
    )


def integrator_comparison(
    dt_days=(0.25, 1.0, 2.0, 4.0), years: float = 200.0, planet: str = "Mercury"
) -> list[dict]:
    """Symplectic against RK4, over long times. The headline of this module.

    A bound orbit integrated with RK4 slowly loses energy and spirals; the
    same orbit with velocity Verlet wobbles and stays. Mercury is used because
    e = 0.206 makes perihelion the hard part.
    """
    a_au, e, m_p = PLANETS[planet]
    a = a_au * AU_M
    mu = G_NEWTON * (M_SUN_KG + m_p)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)
    rows = []
    for dtd in dt_days:
        dt = dtd * DAY_S
        n = int(years * YEAR_S / dt)
        row = {
            "dt_days": dtd,
            "years": years,
            "steps": n,
            "steps_per_orbit": period / dt,
        }
        for method in ("verlet", "rk4"):
            r_p = a * (1.0 - e)
            v_p = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
            tr = integrate(
                [M_SUN_KG, m_p],
                [[0, 0, 0], [r_p, 0, 0]],
                [[0, 0, 0], [0, v_p, 0]],
                dt=dt,
                n_steps=n,
                G=G_NEWTON,
                method=method,
                stride=max(n // 500, 1),
            )
            d = tr.drift(G_NEWTON)
            row[f"{method}_energy_drift"] = d["energy_rel_drift"]
            row[f"{method}_energy_spread"] = d["energy_rel_spread"]
            row[f"{method}_L_drift"] = d["L_rel_drift"]
        rows.append(row)
    return rows
