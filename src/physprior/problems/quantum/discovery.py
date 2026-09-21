"""Recover the spectrum's law from a solved Schrodinger equation.

`spectrum_law`   Give symbolic regression nothing but (n, E_n) from the solver
                 and see whether it returns n^2, (n + 1/2), or -1/n^2. The
                 answer is known exactly, so the residual is the METHOD's
                 error and nothing else.

`schrodinger_vs_nist`  The payoff of having both. The simulation solves the
                 non-relativistic Coulomb problem, which is precisely what
                 Bohr and Schrodinger predict. NIST measures the real atom.
                 The difference between them is not numerical error and is not
                 a fitting artefact -- it is the relativistic and QED
                 correction, the same 10.8 ppm the hydrogen track finds by
                 fitting. Two routes, one number.
"""

from __future__ import annotations

import numpy as np

from physprior.methods.symbolic import fit_sr, power_law_exponent

from . import schrodinger

# cm^-1 per Hartree (CODATA 2022); the solver works in Hartree, NIST in cm^-1.
HARTREE_ICM = 219474.6313632


def spectrum_law(
    system: str = "well", n_levels: int = 10, seed: int = 11, niterations: int = 60
) -> dict:
    """SR on (n, E_n) straight out of the eigenvalue solver."""
    sp = {
        "well": lambda: schrodinger.infinite_well(n_levels=n_levels),
        "harmonic": lambda: schrodinger.harmonic_oscillator(n_levels=n_levels),
        "hydrogen": lambda: schrodinger.hydrogen_radial(n_levels=n_levels),
    }[system]()
    n = sp.n.astype(float)
    E = sp.energy
    scale = float(np.max(np.abs(E)))
    f = fit_sr(
        n.reshape(-1, 1),
        E / scale,
        feature_names=["n"],
        niterations=niterations,
        maxsize=14,
        seed=seed,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["square", "inv"],
    )
    # Measure the power of n, which is the physics: 2, 1 or -2.
    p = power_law_exponent(f, "n", (float(n.min()), float(n.max())), tol=0.03)
    expected = {"well": 2.0, "harmonic": None, "hydrogen": -2.0}[system]
    return {
        "system": system,
        "law": sp.meta["law"],
        "n_levels": n_levels,
        "solver_max_rel_error": sp.max_rel_error,
        "sr_expression": f.expression,
        "scale": scale,
        "exponent_found": p,
        "exponent_expected": expected,
        "exponent_error": (
            None if p is None or expected is None else float(abs(p - expected))
        ),
        # The oscillator is affine in n, not a power law, so the right check
        # there is the spacing: E_{n+1} - E_n must be constant and equal omega.
        "level_spacing": float(np.mean(np.diff(E))),
        "level_spacing_std": float(np.std(np.diff(E))),
        "energies": E.tolist(),
        "n": n.tolist(),
    }


def schrodinger_vs_nist(n_max: int = 20) -> dict:
    """Simulated hydrogen against the real atom. The gap is QED."""
    from physprior.constants import RYDBERG_H_CM
    from physprior.data.sources import nist

    lv = nist.load()
    # Richardson-extrapolated: the plain solver's 300 ppm error would
    # swamp the 10.8 ppm effect this comparison exists to see.
    sp = schrodinger.hydrogen_richardson(n_levels=n_max, r_max=2000.0, n_grid=80000)

    # Transition energies above the ground state, both in cm^-1.
    sim_icm = (sp.energy - sp.energy[0]) * HARTREE_ICM
    keep = np.isin(lv.n, sp.n)
    obs_icm = lv.energy_icm[keep]
    n = lv.n[keep]
    sim_icm = sim_icm[: len(obs_icm)]

    # The infinite-nuclear-mass solver has no reduced-mass factor; put it in,
    # otherwise the comparison is dominated by a known 545 ppm, not by QED.
    from physprior.constants import M_ELECTRON, M_PROTON

    mu_factor = M_PROTON / (M_ELECTRON + M_PROTON)
    sim_icm = sim_icm * mu_factor

    rel = (obs_icm[1:] - sim_icm[1:]) / obs_icm[1:]
    return {
        "n": n.tolist(),
        "observed_icm": obs_icm.tolist(),
        "simulated_icm": sim_icm.tolist(),
        "reduced_mass_factor": mu_factor,
        "mean_gap_ppm": float(np.mean(rel) * 1e6),
        "gap_ppm_by_n": (rel * 1e6).tolist(),
        "solver_max_rel_error": sp.max_rel_error,
        "solver_max_rel_error_ppm": sp.max_rel_error * 1e6,
        "solver_without_richardson_ppm": sp.meta.get(
            "plain_max_rel_error", float("nan")
        )
        * 1e6,
        "nist_limit_icm": lv.limit_icm,
        "bohr_rydberg_icm": RYDBERG_H_CM,
        "limit_minus_bohr_ppm": (lv.limit_icm - RYDBERG_H_CM) / RYDBERG_H_CM * 1e6,
        "note": (
            "the simulation is the non-relativistic Coulomb problem, i.e. "
            "exactly what Bohr/Schrodinger predict; the residual against "
            "NIST is relativistic + QED, not solver error"
        ),
    }
