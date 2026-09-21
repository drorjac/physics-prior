"""Track-level guards: the physics each track exists to demonstrate."""

import numpy as np
import pytest


def test_gravity_pn_ablation_reduces_the_bias():
    """0PN must be biased and 2PN must not. This is track G's headline."""
    from physprior.constants import GW150914_MCHIRP_DETECTOR as PUB
    from physprior.problems.relativity import gw150914 as G

    prob, _ = G.problem()
    rows = {r["pn_order"]: r for r in G.pn_ablation(prob.x, prob.y)}
    newt, pn2 = rows[0], rows[3]
    assert newt["converged"] and pn2["converged"]
    assert abs(newt["bias_sigma"]) > 2.0  # Newtonian is biased
    assert abs(pn2["Mc"] - PUB) < 2.0  # 2PN is not
    assert abs(pn2["bias_sigma"]) < abs(newt["bias_sigma"])


def test_hydrogen_fit_finds_the_ionisation_limit_and_the_qed_gap():
    from physprior.benchmark.protocol import fit_arm
    from physprior.problems.quantum import hydrogen as A

    prob, meta = A.problem()
    f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
    R = f.params["R"]
    assert abs(R - meta["ionisation_limit_icm"]) / R < 1e-6  # < 1 ppm
    assert 10.0 < meta["limit_minus_bohr_ppm"] < 12.0  # QED, ~10.8 ppm


def test_cmb_fit_is_planckian():
    from physprior.benchmark.metrics import chi2_reduced
    from physprior.benchmark.protocol import fit_arm
    from physprior.problems.quantum import cmb as Q

    prob, meta = Q.problem()
    f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
    assert abs(f.params["T"] - meta["published_T_K"]) < 0.002
    assert chi2_reduced(prob.y, f.predict(prob.x), prob.sigma, 1) < 2.0


def test_kepler_gm_sun_within_100_ppm():
    from physprior.benchmark.protocol import fit_arm
    from physprior.constants import GM_SUN
    from physprior.problems.gravity import kepler as R

    prob, _ = R.problem()
    f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
    assert abs(f.params["GM"] - GM_SUN) / GM_SUN < 1e-4


@pytest.mark.slow
def test_mercury_gr_coefficient_converges_to_one():
    """The result is alpha once it has stopped moving, so the guard is on the
    converged stencil, and on the fact that the coarse one is NOT trusted."""
    from physprior.problems.relativity import mercury as R

    coarse = R.gr_regression(step="180m", order=4)
    fine = R.gr_regression(step="180m", order=6)
    assert abs(fine["alpha_GR"] - 1.0) < 0.01
    assert abs(coarse["alpha_GR"] - 1.0) > abs(fine["alpha_GR"] - 1.0)
    lad = fine["residual_ladder"]
    assert lad["sun_only"] > lad["sun_planets"] > lad["sun_planets_gr"]
