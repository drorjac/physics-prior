"""Guards on the simulations.

A simulation is only useful as a control if it is right, so each one is
checked against a closed-form answer, and the numerical discipline the project
preaches is enforced here too: conservation laws, convergence order, and
unitarity.
"""

from __future__ import annotations

import numpy as np
import pytest

from physprior.constants import G_NEWTON

# --------------------------------------------------------------------- gravity


def test_verlet_beats_rk4_over_long_times():
    """Symplectic integration is the point of `integrators.py`."""
    from physprior.problems.gravity import orbits as sim

    rows = {
        r["dt_days"]: r for r in sim.integrator_comparison(dt_days=(4.0,), years=200.0)
    }
    r = rows[4.0]
    assert r["verlet_energy_drift"] < 0.1  # bounded
    assert r["rk4_energy_drift"] > 1.0  # the orbit unbinds
    assert r["rk4_energy_drift"] > 10 * r["verlet_energy_drift"]


def test_two_body_conserves_energy_and_angular_momentum():
    from physprior.problems.gravity import orbits as sim

    tr = sim.two_body("Mercury", n_orbits=3.0, steps_per_orbit=2000, stride=4)
    d = tr.drift(G_NEWTON)
    assert d["energy_rel_drift"] < 1e-10
    assert d["L_rel_drift"] < 1e-10


def test_two_body_reproduces_the_intended_ellipse():
    """`a` and `e` must mean what they usually mean."""
    from physprior.problems.gravity import orbits as sim

    tr = sim.two_body("Mercury", n_orbits=2.0, steps_per_orbit=4000, stride=1)
    r = np.linalg.norm(tr.r[:, 1] - tr.r[:, 0], axis=1)
    a, e = tr.meta["a_m"], tr.meta["ecc"]
    assert abs(r.min() - a * (1 - e)) / (a * (1 - e)) < 1e-4
    assert abs(r.max() - a * (1 + e)) / (a * (1 + e)) < 1e-4


def test_figure_eight_is_periodic():
    from physprior.problems.gravity import orbits as sim

    tr = sim.three_body("figure8", n_periods=1.0, steps_per_period=8000, stride=4)
    closure = np.linalg.norm(tr.r[-1] - tr.r[0])
    assert closure < 1e-4, f"choreography did not close: {closure:.2e}"


def test_three_body_is_measurably_chaotic():
    from physprior.problems.gravity import orbits as sim

    ly = sim.lyapunov_separation(1e-9, n_periods=10.0)
    assert ly["lambda"] > 0.01, "no exponential separation -- not chaotic"
    assert ly["separation"][-1] > 100 * ly["separation"][0]


@pytest.mark.sr
def test_force_law_recovered_from_simulated_orbit():
    """The controlled twin of the real-data tracks: truth is exactly -2."""
    from physprior.problems.gravity import discovery as discover

    d = discover.force_law("Mercury", n_orbits=2.0, steps_per_orbit=3000)
    assert abs(d["exponent_loglog"] + 2.0) < 1e-4
    assert d["exponent_sr"] is not None
    assert abs(d["exponent_sr"] + 2.0) < 1e-3
    # the direct estimator must beat the log-log intercept by a wide margin
    assert abs(d["mu_direct_rel_error_ppb"]) < abs(d["mu_loglog_rel_error_ppm"]) * 1000


def test_law_survives_chaos():
    """Chaos is a property of the solution, not of the equation."""
    from physprior.problems.gravity import discovery as discover

    c = discover.chaotic_law(n_periods=4.0, steps_per_period=4000)
    assert abs(c["Gm_recovered"] - 1.0) < 1e-4
    assert c["lyapunov"] > 0.01


# ------------------------------------------------------------------ relativity


def test_mercury_precession_matches_einstein():
    from physprior.problems.relativity import spacetime as sim

    mp = sim.mercury_precession(n_orbits=5)
    assert abs(mp["precession_arcsec_per_century"] - 42.98) < 0.05
    # the Newtonian control must be small compared with the signal
    assert mp["control_fraction_of_signal"] < 0.02


def test_newtonian_orbit_does_not_precess():
    from physprior.problems.relativity import spacetime as sim

    run = sim.schwarzschild_orbit(n_orbits=5.0, relativistic=False)
    gr = sim.schwarzschild_orbit(n_orbits=5.0, relativistic=True)
    assert abs(run.precession_per_orbit) < 0.05 * abs(gr.precession_per_orbit)


def test_light_deflection_is_twice_newtonian():
    from physprior.problems.relativity import spacetime as sim

    ld = sim.light_deflection()
    assert abs(ld["deflection_gr_arcsec"] - 1.7512) < 1e-3
    assert abs(ld["deflection_numeric_arcsec"] - ld["deflection_gr_arcsec"]) < 0.02
    assert (
        abs(ld["deflection_gr_arcsec"] / ld["deflection_newtonian_arcsec"] - 2.0) < 1e-9
    )


def test_inspiral_chirp_has_a_ringdown_and_peaks_at_merger():
    """Without the ringdown the envelope peaks mid-inspiral and the injection
    silently discards its own highest-frequency cycles."""
    from physprior.problems.relativity import spacetime as sim

    wf = sim.inspiral_chirp(31.17, ringdown=True)
    assert wf["t"][-1] > 0.0  # tail after the merger
    i_peak = int(np.argmax(wf["amplitude"]))
    assert abs(wf["t"][i_peak]) < 2e-3  # amplitude peaks at merger
    assert wf["f"][0] < wf["f"][int(len(wf["f"]) * 0.5)]  # it chirps up


def test_gr_coefficient_recovered_from_simulated_orbit():
    from physprior.problems.relativity import discovery as discover

    d = discover.precession_recovery(n_orbits=5)
    assert abs(d["alpha_recovered"] - 1.0) < 0.02


# --------------------------------------------------------------------- quantum


@pytest.mark.parametrize(
    "maker,tol",
    [
        ("infinite_well", 2e-5),
        ("harmonic_oscillator", 2e-5),
    ],
)
def test_textbook_spectra(maker, tol):
    from physprior.problems.quantum import schrodinger as sim

    sp = getattr(sim, maker)(n_levels=8)
    assert sp.max_rel_error < tol


def test_hydrogen_levels_and_richardson():
    from physprior.problems.quantum import schrodinger as sim

    plain = sim.hydrogen_radial(n_levels=6, n_grid=80000)
    rich = sim.hydrogen_richardson(n_levels=6, n_grid=80000)
    assert rich.max_rel_error < plain.max_rel_error
    assert rich.max_rel_error < 2e-5  # below the 10.8 ppm QED signal


def test_finite_differences_are_second_order():
    """Halving dx must quarter the error, or the solver is not what we think."""
    from physprior.problems.quantum import schrodinger as sim

    rows = sim.grid_convergence(grids=(500, 1000, 2000, 4000), n_levels=4)
    ratios = [r["error_ratio_vs_coarser"] for r in rows[1:]]
    assert all(3.0 < r < 5.0 for r in ratios), ratios


def test_inner_boundary_matters_more_than_step_size():
    from physprior.problems.quantum import schrodinger as sim

    rows = {
        r["r_min"]: r
        for r in sim.r_min_sweep(r_mins=(1e-6, 1e-3), n_grid=(40000, 80000))
    }
    fine, coarse = rows[1e-6], rows[1e-3]
    # refining the grid helps at small r_min ...
    assert fine["max_rel_err_n80000"] < 0.6 * fine["max_rel_err_n40000"]
    # ... and does essentially nothing at large r_min
    assert coarse["max_rel_err_n80000"] > 0.9 * coarse["max_rel_err_n40000"]


def test_split_operator_is_unitary():
    from physprior.problems.quantum import schrodinger as sim

    wp = sim.wavepacket(n_steps=300)
    assert wp["norm_drift"] < 1e-12
    assert not wp["classically_allowed"]  # genuinely tunnelling
    assert 0.0 < wp["transmission"] < 1.0


@pytest.mark.sr
def test_spectrum_laws_recovered():
    from physprior.problems.quantum import discovery as discover

    well = discover.spectrum_law("well", n_levels=10)
    assert abs(well["exponent_found"] - 2.0) < 1e-3
    hyd = discover.spectrum_law("hydrogen", n_levels=10)
    assert abs(hyd["exponent_found"] + 2.0) < 1e-3
    sho = discover.spectrum_law("harmonic", n_levels=10)
    # affine in n, so NOT a power law; the spacing is the right statistic
    assert sho["exponent_found"] is None
    assert abs(sho["level_spacing"] - 1.0) < 1e-4
    assert sho["level_spacing_std"] < 1e-4


@pytest.mark.slow
def test_injection_recovers_the_injected_chirp_mass():
    """The pipeline's own error budget. Without this the GW150914 number is
    a number with no uncertainty attached to the method that produced it."""
    from physprior.problems.relativity import discovery as discover

    r = discover.injection_test(seeds=(11, 23, 42, 101, 202, 303))
    assert r["n_usable"] >= 5
    assert abs(r["pn3"]["bias_pct"]) < 10.0  # 2PN is unbiased
    assert r["pn0"]["bias_pct"] > 10.0  # Newtonian is not
