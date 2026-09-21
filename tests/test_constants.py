"""The constants are the yardstick. If they are wrong, every result is."""

import math

from physprior import constants as k


def test_rydberg_hydrogen_matches_codata():
    # R_H = R_inf * m_p/(m_e+m_p) = 109677.583 cm^-1
    assert abs(k.RYDBERG_H_CM - 109677.583) < 0.01


def test_solar_mass_in_seconds():
    assert abs(k.T_SUN_S - 4.925491e-6) < 1e-11


def test_detector_frame_chirp_mass():
    assert abs(k.GW150914_MCHIRP_DETECTOR - 28.6 * 1.09) < 1e-9


def test_h_over_k():
    assert abs(k.H_OVER_K - k.H_PLANCK / k.K_BOLTZMANN) < 1e-30


def test_arcsec_per_rad():
    assert abs(k.ARCSEC_PER_RAD - 206264.806) < 0.01


def test_constants_module_defines_no_guard():
    """The guards live in `physprior.units` and raise `UnitError`.

    They used to be here and to raise `AssertionError`, which `python -O`
    strips. Keeping them out of this module is the point.
    """
    assert not hasattr(k, "assert_unit")
    assert not hasattr(k, "require")


def test_gm_sun_consistent_with_mass_and_G():
    assert math.isclose(k.GM_SUN, k.G_NEWTON * k.M_SUN_KG, rel_tol=1e-12)
