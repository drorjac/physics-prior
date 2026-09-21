"""Loaders: shapes, units, and the physics that must be in the numbers.

Every test here reads a downloaded file, so all of them are marked `network`:
on a cold cache they fetch, and with `PHYSPRIOR_OFFLINE=1` they skip. CI runs
without them, which is what keeps CI hermetic.
"""

import numpy as np
import pytest

from physprior.constants import RYDBERG_H_CM

pytestmark = pytest.mark.network


def test_firas_shape_and_units():
    from physprior.data.sources import firas

    s = firas.load()
    assert len(s) == 43
    assert 6.8e10 < s.nu_hz.min() < 6.9e10  # 68 GHz
    assert 6.3e11 < s.nu_hz.max() < 6.4e11  # 639 GHz
    # peak of a 2.725 K blackbody is ~384 MJy/sr at ~160 GHz
    assert 380 < s.intensity.max() < 390
    assert (s.sigma > 0).all() and s.sigma.max() / s.intensity.max() < 1e-2


def test_hydrogen_levels_are_bohr_to_ppm():
    from physprior.data.sources import nist as hydrogen

    h = hydrogen.load()
    assert h.n.min() == 1 and h.n.max() >= 30
    assert np.all(np.diff(h.energy_icm) > 0)
    # Bohr must hold to about 20 ppm; it is exactly the failure at 10.8 ppm
    # that track A is about, so this is a loose guard, not a tight one.
    pred = RYDBERG_H_CM * (1 - 1 / h.n[1:] ** 2)
    rel = np.abs(h.energy_icm[1:] - pred) / h.limit_icm
    assert rel.max() < 2e-5
    assert 109678.7 < h.limit_icm < 109678.8


def test_gw_chirp_track_is_a_chirp():
    from physprior.data.sources import gwosc as gw

    tr = gw.frequency_track()
    assert 5 <= len(tr) <= 20
    assert tr.f_hz.min() > 30 and tr.f_hz.max() < 250
    # the merger must land within 40 ms of the GWOSC event time
    assert abs(tr.t_peak_s) < 0.04
    # every retained cycle is before the merger, and the frequency rises
    assert (tr.t_s <= tr.t_peak_s + 1e-9).all()
    assert np.polyfit(tr.t_s, tr.f_hz, 1)[0] > 0


def test_gw_strain_whitening_is_unit_ish():
    from physprior.data.sources import gwosc as gw

    t, h, fs, _ = gw.conditioned()
    off = (np.abs(t) > 1.0) & (t > t[0] + 2) & (t < t[-1] - 2)
    assert 0.05 < np.std(h[off]) < 5.0  # whitened, band-limited
    assert fs == 4096.0


@pytest.mark.parametrize(
    "name,a_lo,a_hi", [("Mercury", 0.38, 0.39), ("Neptune", 30.0, 30.4)]
)
def test_planet_elements(name, a_lo, a_hi):
    from physprior.data.sources import horizons as ephemeris

    pt = ephemeris.planets()
    i = pt.name.index(name)
    assert a_lo < pt.a_au[i] < a_hi
    # Kepler's third law must hold across the table to better than 0.2%
    k = pt.period_d**2 / pt.a_au**3
    assert (k.max() - k.min()) / k.mean() < 2e-3
