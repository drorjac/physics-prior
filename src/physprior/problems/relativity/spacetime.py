"""Relativity simulations.

`schwarzschild_orbit`  The orbit equation for a test particle around a
                       Schwarzschild mass, in the standard u = 1/r form:

                           d2u/dphi2 + u = GM/h^2 + (3GM/c^2) u^2

                       The last term is the whole of general relativity as far
                       as planetary orbits are concerned. Set it to zero and
                       the orbit is a closed ellipse; switch it on and the
                       ellipse precesses. Per orbit the shift is

                           dphi = 6 pi GM / (c^2 a (1 - e^2))

                       which for Mercury is 43 arcsec per century. The
                       simulation is run BOTH ways, so the precession is
                       measured as a difference rather than asserted.

`inspiral_chirp`       A post-Newtonian binary inspiral: frequency, phase and
                       strain h(t) for given chirp mass. This is the model
                       behind the GW150914 track, run forward so the real
                       whitened strain can be laid over a simulated one.

`light_deflection`     A photon's null geodesic past a mass, giving the
                       1.75 arcsec at the solar limb that made Einstein famous
                       in 1919.

The relativistic factor is tiny for real systems (3GM/(c^2 r) ~ 1e-7 for
Mercury), so every function takes `gr_boost`. At `gr_boost = 1` the physics is
real and the effect is invisible on a plot; at 1e5 it is obvious and wrong.
Both are useful and the value is always recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from physprior.constants import (
    ARCSEC_PER_RAD,
    AU_M,
    C_LIGHT,
    GM_SUN,
    JULIAN_CENTURY_D,
    T_SUN_S,
)


@dataclass
class OrbitRun:
    phi: np.ndarray  # true anomaly, rad
    r: np.ndarray  # radius, m
    x: np.ndarray  # Cartesian, m
    y: np.ndarray
    perihelion_phi: np.ndarray  # phi at each perihelion passage, rad
    precession_per_orbit: float  # rad
    meta: dict = field(default_factory=dict)


def schwarzschild_orbit(
    a_m: float = 0.387098 * AU_M,
    ecc: float = 0.205630,
    gm: float = GM_SUN,
    n_orbits: float = 3.0,
    gr_boost: float = 1.0,
    points_per_orbit: int = 4000,
    relativistic: bool = True,
) -> OrbitRun:
    """Integrate d2u/dphi2 + u = GM/h^2 + gr_boost * (3GM/c^2) u^2.

    Started at perihelion, where u = 1/r_p and du/dphi = 0. The specific
    angular momentum h comes from the Newtonian ellipse, h^2 = GM a (1-e^2),
    so `a` and `e` mean what they usually mean.
    """
    h2 = gm * a_m * (1.0 - ecc**2)
    eps = gr_boost * 3.0 * gm / C_LIGHT**2

    def rhs(_phi, s):
        u, du = s
        return [du, gm / h2 - u + (eps * u**2 if relativistic else 0.0)]

    u0 = 1.0 / (a_m * (1.0 - ecc))
    phi_end = 2.0 * np.pi * n_orbits + 0.6 * np.pi
    phi = np.linspace(0.0, 2.0 * np.pi * n_orbits, int(points_per_orbit * n_orbits))
    sol = solve_ivp(
        rhs,
        (0.0, phi_end),
        [u0, 0.0],
        t_eval=None,
        rtol=1e-12,
        atol=1e-20,
        method="DOP853",
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"orbit integration failed: {sol.message}")
    u = sol.sol(phi)[0]
    r = 1.0 / u

    # Perihelion is where du/dphi changes sign from + to -. Finding that root
    # on the DENSE solution, rather than fitting a parabola to the sampled
    # grid, is what makes this measurement usable at all: Mercury's shift is
    # 5e-7 rad per orbit, and a grid-and-parabola estimate is only good to
    # ~1e-6 rad -- bigger than the effect. Root-finding gets ~1e-9.
    peri = []
    for k in range(1, int(n_orbits) + 1):
        lo, hi = 2.0 * np.pi * k - 0.8 * np.pi, 2.0 * np.pi * k + 0.5 * np.pi
        if hi > phi_end:
            break
        try:
            peri.append(
                brentq(lambda q: sol.sol(q)[1], lo, hi, xtol=1e-14, rtol=8.9e-16)
            )
        except ValueError:
            break
    peri_arr = np.asarray(peri, dtype=float)
    # No perihelion found means no measurement: NaN, not a zero shift.
    prec = (
        float((peri_arr[-1] - 2.0 * np.pi * len(peri_arr)) / len(peri_arr))
        if len(peri_arr)
        else float("nan")
    )

    return OrbitRun(
        phi=phi,
        r=r,
        x=r * np.cos(phi),
        y=r * np.sin(phi),
        perihelion_phi=peri_arr,
        precession_per_orbit=prec,
        meta={
            "a_m": a_m,
            "ecc": ecc,
            "gm": gm,
            "gr_boost": gr_boost,
            "relativistic": relativistic,
            "n_orbits": n_orbits,
            "h2": h2,
            "eps": eps,
            "analytic_precession_per_orbit": 6.0
            * np.pi
            * gm
            * gr_boost
            / (C_LIGHT**2 * a_m * (1 - ecc**2)),
        },
    )


def mercury_precession(n_orbits: float = 5.0, points_per_orbit: int = 2000) -> dict:
    """Mercury, integrated with and without the GR term.

    The Newtonian run is the control: a Kepler ellipse does not precess, so
    whatever drift it shows is numerical, and it is reported rather than
    hidden. The quoted GR precession is the DIFFERENCE of the two runs, which
    cancels that common bias -- the same trick the ephemeris track uses when
    it subtracts the planetary perturbations.
    """
    a, e = 0.387098 * AU_M, 0.205630
    period_d = 87.9691
    per_century = JULIAN_CENTURY_D / period_d

    gr = schwarzschild_orbit(
        a, e, n_orbits=n_orbits, points_per_orbit=points_per_orbit, relativistic=True
    )
    newt = schwarzschild_orbit(
        a, e, n_orbits=n_orbits, points_per_orbit=points_per_orbit, relativistic=False
    )
    d_gr = gr.precession_per_orbit - newt.precession_per_orbit
    return {
        "precession_per_orbit_rad": d_gr,
        "precession_arcsec_per_century": d_gr * ARCSEC_PER_RAD * per_century,
        "analytic_arcsec_per_century": gr.meta["analytic_precession_per_orbit"]
        * ARCSEC_PER_RAD
        * per_century,
        "newtonian_control_arcsec_per_century": newt.precession_per_orbit
        * ARCSEC_PER_RAD
        * per_century,
        "n_orbits": n_orbits,
        "points_per_orbit": points_per_orbit,
        "orbital_period_days": period_d,
        "gr_precession_per_orbit_rad": gr.precession_per_orbit,
        "newtonian_precession_per_orbit_rad": newt.precession_per_orbit,
        "control_fraction_of_signal": abs(
            newt.precession_per_orbit / gr.precession_per_orbit
        ),
    }


def inspiral_chirp(
    mchirp_msun: float = 31.17,
    eta: float = 0.2486,
    f_start: float = 35.0,
    f_end: float = 250.0,
    fs: float = 4096.0,
    pn_order: int = 3,
    ringdown: bool = True,
    tau_rd: float = 4.0e-3,
) -> dict:
    """A post-Newtonian inspiral: f(t), phase and the strain h(t).

    Amplitude follows the Newtonian quadrupole scaling, h ~ f^(2/3), which is
    all that is needed to compare the SHAPE against real whitened strain --
    whitening removes the absolute scale anyway.

    `ringdown` appends a short exponentially damped tail at the final
    frequency. It is not cosmetic. Without it the waveform stops dead at
    f_end, and the smoothed envelope of the band-passed result then peaks in
    the MIDDLE of the inspiral rather than at the merger. Since the track's
    extraction keeps only the cycles before the envelope peak, an injection
    without a ringdown silently throws away its own highest-frequency cycles
    and comes back with a chirp mass biased by more than half. That was found
    by injecting a known mass and getting 13.7 back instead of 31.2.
    """
    from physprior.problems.relativity.gw150914 import (
        fdot_np,  # the same law the track fits
    )

    f = np.linspace(f_start, f_end, 200000)
    dt_df = 1.0 / fdot_np(f, mchirp_msun, eta, pn_order)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (dt_df[1:] + dt_df[:-1]) * np.diff(f))])
    n = int(t[-1] * fs)
    tt = np.linspace(0.0, t[-1], max(n, 64))
    ff = np.interp(tt, t, f)
    phase = (
        2.0
        * np.pi
        * np.concatenate([[0.0], np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(tt))])
    )
    amp = (ff / f_start) ** (2.0 / 3.0)
    t_rel = tt - tt[-1]
    h = amp * np.cos(phase)

    if ringdown:
        n_rd = int(6.0 * tau_rd * fs)
        t_rd = np.arange(1, n_rd + 1) / fs
        env_rd = amp[-1] * np.exp(-t_rd / tau_rd)
        h_rd = env_rd * np.cos(phase[-1] + 2.0 * np.pi * f_end * t_rd)
        t_rel = np.concatenate([t_rel, t_rd])
        h = np.concatenate([h, h_rd])
        ff = np.concatenate([ff, np.full(n_rd, f_end)])
        amp = np.concatenate([amp, env_rd])

    return {
        "t": t_rel,
        "f": ff,
        "phase": phase,
        "h": h,
        "amplitude": amp,
        "ringdown": ringdown,
        "t_merger_s": 0.0,
        "duration_s": float(t[-1]),
        "mchirp_msun": mchirp_msun,
        "eta": eta,
        "pn_order": pn_order,
        "mchirp_seconds": mchirp_msun * T_SUN_S,
    }


def light_deflection(impact_parameter_m: float = 6.957e8, gm: float = GM_SUN) -> dict:
    """Null geodesic deflection. GR gives 4GM/(c^2 b), twice the Newtonian
    value, which is what the 1919 eclipse measured."""
    b = impact_parameter_m
    gr = 4.0 * gm / (C_LIGHT**2 * b)
    # Numerical check from the photon orbit equation d2u/dphi2 + u = 3GM u^2/c^2.
    # The photon comes in from infinity (u = 0) with du/dphi = 1/b and leaves
    # when u returns to 0; the total sweep exceeds pi by the deflection.
    eps = 3.0 * gm / C_LIGHT**2

    def rhs(_p, y):
        u, du = y
        return [du, eps * u**2 - u]

    phi = np.linspace(0.0, 1.3 * np.pi, 400000)
    sol = solve_ivp(
        rhs,
        (phi[0], phi[-1]),
        [0.0, 1.0 / b],
        t_eval=phi,
        rtol=1e-12,
        atol=1e-16,
        method="DOP853",
    )
    u = sol.y[0]
    # first index past the peak where u crosses back through zero
    peak = int(np.argmax(u))
    after = np.where(u[peak:] <= 0.0)[0]
    if len(after):
        i = peak + int(after[0])
        # linear interpolation of the crossing
        phi_end = phi[i - 1] + (0.0 - u[i - 1]) * (phi[i] - phi[i - 1]) / (
            u[i] - u[i - 1]
        )
        numeric = float(phi_end - np.pi)
    else:
        numeric = float("nan")
    return {
        "impact_parameter_m": b,
        "deflection_gr_arcsec": gr * ARCSEC_PER_RAD,
        "deflection_newtonian_arcsec": 0.5 * gr * ARCSEC_PER_RAD,
        "deflection_numeric_arcsec": numeric * ARCSEC_PER_RAD,
        "solar_limb": abs(b - 6.957e8) < 1e6,
    }
