"""Quantum simulations: solve the Schrodinger equation, then rediscover the
spectrum's law from the numbers that come out.

Everything is in atomic-like units, hbar = m = 1, so the answers are the ones
from the textbook:

    infinite well, width L     E_n = n^2 pi^2 / (2 L^2)          E ~ n^2
    harmonic oscillator        E_n = (n + 1/2) omega             E ~ n
    hydrogen (radial, l = 0)   E_n = -1 / (2 n^2)                E ~ -1/n^2

`solve_1d` is a second-order finite-difference discretisation of

    -1/2 psi'' + V(x) psi = E psi

on a uniform grid with psi = 0 at both ends, diagonalised directly. It is the
first numerical method in a quantum course and it is enough: with a few
thousand grid points the low-lying levels are accurate to 1e-6 or better, and
its error is O(dx^2), which the convergence helper measures rather than
assumes.

`wavepacket` evolves a Gaussian through a barrier with the split-operator
method, which is unitary by construction -- the norm is conserved to machine
precision, and that is checked, not hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal

from physprior.units import require_not_none


@dataclass
class Spectrum:
    n: np.ndarray  # quantum number, 1-based for wells, 0-based for SHO
    energy: np.ndarray  # eigenvalues, Hartree-like units
    psi: np.ndarray  # (n_levels, n_grid) normalised eigenfunctions
    x: np.ndarray
    V: np.ndarray
    exact: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def max_rel_error(self) -> float:
        if self.exact is None:
            return float("nan")
        exact = self.exact
        if exact is None:
            return float("nan")
        return float(np.max(np.abs((self.energy - exact) / exact)))


def solve_1d(
    potential,
    x_min: float,
    x_max: float,
    n_grid: int = 4000,
    n_levels: int = 12,
    n_offset: int = 1,
    exact=None,
    meta: dict | None = None,
) -> Spectrum:
    """Eigenvalues of -1/2 psi'' + V psi on [x_min, x_max], psi(ends) = 0."""
    x = np.linspace(x_min, x_max, n_grid)
    dx = x[1] - x[0]
    V = np.asarray(potential(x), float)
    interior = slice(1, -1)
    d = 1.0 / dx**2 + V[interior]
    e = -0.5 / dx**2 * np.ones(len(d) - 1)
    vals, vecs = eigh_tridiagonal(d, e, select="i", select_range=(0, n_levels - 1))

    psi = np.zeros((n_levels, n_grid))
    psi[:, interior] = vecs.T
    psi /= np.sqrt(np.sum(psi**2, axis=1, keepdims=True) * dx)

    n = np.arange(n_levels) + n_offset
    ex = np.asarray(exact(n), float) if callable(exact) else exact
    return Spectrum(
        n=n,
        energy=vals,
        psi=psi,
        x=x,
        V=V,
        exact=ex,
        meta={
            "n_grid": n_grid,
            "dx": dx,
            "x_min": x_min,
            "x_max": x_max,
            **(meta or {}),
        },
    )


# --------------------------------------------------------------------------
# the three textbook potentials
# --------------------------------------------------------------------------


def infinite_well(
    length: float = 1.0, n_levels: int = 12, n_grid: int = 4000
) -> Spectrum:
    """E_n = n^2 pi^2 / (2 L^2). The cleanest E ~ n^2 in physics."""
    return solve_1d(
        lambda x: np.zeros_like(x),
        0.0,
        length,
        n_grid,
        n_levels,
        n_offset=1,
        exact=lambda n: n**2 * np.pi**2 / (2.0 * length**2),
        meta={
            "system": "infinite square well",
            "L": length,
            "law": "E_n = n^2 pi^2 / (2 L^2)",
        },
    )


def harmonic_oscillator(
    omega: float = 1.0, n_levels: int = 12, n_grid: int = 6000, half_width: float = 12.0
) -> Spectrum:
    """E_n = (n + 1/2) omega. Equally spaced levels -- the signature."""
    return solve_1d(
        lambda x: 0.5 * omega**2 * x**2,
        -half_width,
        half_width,
        n_grid,
        n_levels,
        n_offset=0,
        exact=lambda n: (n + 0.5) * omega,
        meta={
            "system": "harmonic oscillator",
            "omega": omega,
            "law": "E_n = (n + 1/2) omega",
        },
    )


def hydrogen_radial(
    l: int = 0, n_levels: int = 8, n_grid: int = 40000, r_max: float = 900.0
) -> Spectrum:
    """Radial Schrodinger for hydrogen: E_n = -1/(2 n^2) Hartree.

    The substitution u = r R turns the 3-D radial equation into a 1-D problem
    with the centrifugal term folded into the potential. `r_max` has to be
    large because the n-th state extends to about 2 n^2 Bohr radii; too small
    a box pushes the high levels up, which is the most common way to get this
    wrong and is exactly what `box_convergence` measures.
    """

    def V(r):
        with np.errstate(divide="ignore"):
            return -1.0 / r + (l * (l + 1)) / (2.0 * r**2)

    return solve_1d(
        V,
        1e-6,
        r_max,
        n_grid,
        n_levels,
        n_offset=l + 1,
        exact=lambda n: -1.0 / (2.0 * n**2),
        meta={
            "system": "hydrogen radial",
            "l": l,
            "r_max_bohr": r_max,
            "law": "E_n = -1/(2 n^2) Hartree",
        },
    )


# --------------------------------------------------------------------------
# convergence -- the same discipline the Mercury track had to learn
# --------------------------------------------------------------------------


def _exact(spectrum: Spectrum) -> np.ndarray:
    """The closed-form eigenvalues, required by the convergence studies."""
    return require_not_none(
        spectrum.exact,
        f"{spectrum.meta.get('system', 'this spectrum')} has no closed-form "
        "eigenvalues, so its error cannot be measured",
    )


def grid_convergence(
    system: str = "harmonic", grids=(500, 1000, 2000, 4000, 8000), n_levels: int = 6
) -> list[dict]:
    """Second-order finite differences: halving dx should quarter the error."""
    rows = []
    for g in grids:
        sp = {
            "harmonic": lambda _g=g: harmonic_oscillator(n_levels=n_levels, n_grid=_g),
            "well": lambda _g=g: infinite_well(n_levels=n_levels, n_grid=_g),
        }[system]()
        rows.append(
            {
                "n_grid": g,
                "dx": sp.meta["dx"],
                "max_rel_error": sp.max_rel_error,
                "E0": float(sp.energy[0]),
                "E0_exact": float(_exact(sp)[0]),
            }
        )
    for a, b in zip(rows[:-1], rows[1:]):
        b["error_ratio_vs_coarser"] = (
            a["max_rel_error"] / b["max_rel_error"] if b["max_rel_error"] else np.nan
        )
    return rows


def box_convergence(
    r_maxes=(100.0, 200.0, 400.0, 900.0), n_levels: int = 8, n_grid: int = 40000
) -> list[dict]:
    """How big must the box be before hydrogen's high levels stop moving?"""
    rows = []
    for rm in r_maxes:
        sp = hydrogen_radial(n_levels=n_levels, n_grid=n_grid, r_max=rm)
        rows.append(
            {
                "r_max_bohr": rm,
                "max_rel_error": sp.max_rel_error,
                "worst_level_n": int(
                    sp.n[int(np.argmax(np.abs((sp.energy - _exact(sp)) / _exact(sp))))]
                ),
                "E_last": float(sp.energy[-1]),
                "E_last_exact": float(_exact(sp)[-1]),
            }
        )
    return rows


# --------------------------------------------------------------------------
# time dependence: a wavepacket tunnelling
# --------------------------------------------------------------------------


def wavepacket(
    barrier_height: float = 1.2,
    barrier_width: float = 0.6,
    k0: float = 1.4,
    x0: float = -14.0,
    sigma: float = 2.0,
    x_half: float = 40.0,
    n_grid: int = 2048,
    t_max: float = 26.0,
    n_steps: int = 600,
) -> dict:
    """Split-operator TDSE. A Gaussian hits a barrier and partly goes through.

    The kinetic and potential parts are applied alternately, each exactly, so
    the evolution is unitary: `norm_drift` below is a machine-precision check,
    not a tolerance.
    """
    x = np.linspace(-x_half, x_half, n_grid, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, dx)
    V = np.where(np.abs(x) < barrier_width / 2.0, barrier_height, 0.0)

    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2) + 1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    dt = t_max / n_steps
    expV = np.exp(-0.5j * V * dt)
    expK = np.exp(-0.5j * k**2 * dt)

    out = np.empty((n_steps + 1, n_grid), complex)
    out[0] = psi
    for i in range(n_steps):
        psi = expV * psi
        psi = np.fft.ifft(expK * np.fft.fft(psi))
        psi = expV * psi
        out[i + 1] = psi

    norm = np.sum(np.abs(out) ** 2, axis=1) * dx
    right = x > barrier_width / 2.0
    trans = float(np.sum(np.abs(out[-1][right]) ** 2) * dx)
    energy = 0.5 * k0**2
    return {
        "x": x,
        "t": np.linspace(0.0, t_max, n_steps + 1),
        "psi": out,
        "V": V,
        "norm_drift": float(abs(norm[-1] - norm[0])),
        "transmission": trans,
        "energy": energy,
        "barrier_height": barrier_height,
        "barrier_width": barrier_width,
        "classically_allowed": bool(energy > barrier_height),
        "meta": {"k0": k0, "sigma": sigma, "n_grid": n_grid, "dt": dt},
    }


def hydrogen_richardson(
    l: int = 0,
    n_levels: int = 8,
    n_grid: int = 80000,
    r_max: float = 900.0,
    r_min: float = 1e-6,
) -> Spectrum:
    """Hydrogen levels, Richardson-extrapolated to remove the leading grid error.

    Why this exists. Plain second-order finite differences on a uniform grid
    give the hydrogen levels to about 3e-4 relative. That is 300 ppm, and the
    thing worth looking for in this atom -- the relativistic and QED shift the
    NIST data carries -- is 10.8 ppm. A solver 30 times less accurate than the
    effect cannot see it, and reporting the difference anyway would be
    reporting discretisation error as physics.

    Two runs, at dx and dx/2, combined as E + (E_fine - E_coarse)/3, cancel
    the leading term and reach about 4 ppm. That is finally smaller than the
    signal.

    Two traps, both measured in `box_convergence` and in the r_min sweep:
      * `r_max` must hold the n-th state, which reaches ~2 n^2 Bohr radii;
      * `r_min` must be SMALL. At r_min = 1e-3 the error stalls at 4e-3 and
        refining the grid does not help at all -- the inner boundary, not the
        step size, is the limit.
    """
    coarse = hydrogen_radial(l=l, n_levels=n_levels, n_grid=n_grid, r_max=r_max)
    fine = hydrogen_radial(l=l, n_levels=n_levels, n_grid=2 * n_grid, r_max=r_max)
    energy = fine.energy + (fine.energy - coarse.energy) / 3.0
    return Spectrum(
        n=fine.n,
        energy=energy,
        psi=fine.psi,
        x=fine.x,
        V=fine.V,
        exact=fine.exact,
        meta={
            **fine.meta,
            "richardson": True,
            "n_grid_coarse": n_grid,
            "n_grid_fine": 2 * n_grid,
            "plain_max_rel_error": fine.max_rel_error,
            "system": "hydrogen radial (Richardson)",
        },
    )


def r_min_sweep(
    r_mins=(1e-6, 1e-5, 1e-4, 1e-3), n_grid=(40000, 80000, 160000), n_levels: int = 6
) -> list[dict]:
    """The inner boundary matters more than the step size. Shown, not claimed."""
    rows = []
    for rm in r_mins:
        row = {"r_min": rm}
        for ng in n_grid:

            def V(r):
                with np.errstate(divide="ignore"):
                    return -1.0 / r

            sp = solve_1d(
                V,
                rm,
                900.0,
                ng,
                n_levels,
                n_offset=1,
                exact=lambda n: -1.0 / (2.0 * n**2),
            )
            row[f"max_rel_err_n{ng}"] = sp.max_rel_error
        rows.append(row)
    return rows
