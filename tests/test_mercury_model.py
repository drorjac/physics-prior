"""The n-body relativistic term must contain the one-body one as its limit.

`_eih_1pn` is what closes Mercury's residual, so it is checked against the
one physics statement that pins it without any data: with a static central
mass and a massless test body, EIH is exactly the Schwarzschild 1PN term in
harmonic coordinates, which is the `a_gr` column the track has always used.
"""

from __future__ import annotations

import numpy as np

from physprior.constants import C_LIGHT, GM_SUN
from physprior.problems.relativity.mercury import _eih_1pn


def _schwarzschild(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    rn = np.linalg.norm(r, axis=1)[:, None]
    vn2 = np.sum(v * v, axis=1)[:, None]
    rv = np.sum(r * v, axis=1)[:, None]
    return (GM_SUN / (C_LIGHT**2 * rn**3)) * ((4 * GM_SUN / rn - vn2) * r + 4 * rv * v)


def _mercury_like_states(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    r = rng.normal(size=(n, 3)) * 4e10 + np.array([5.8e10, 0.0, 0.0])
    v = rng.normal(size=(n, 3)) * 1.5e4 + np.array([0.0, 4.7e4, 0.0])
    return r, v


def test_eih_reduces_to_schwarzschild_for_a_test_body():
    r, v = _mercury_like_states()
    zero = np.zeros_like(r)
    R = np.array([zero, r])
    V = np.array([zero, v])
    a1 = _eih_1pn(R, V, np.array([GM_SUN, 0.0]))
    # The Sun feels nothing from a massless body; the body feels Schwarzschild.
    assert np.allclose(a1[0], 0.0)
    np.testing.assert_allclose(a1[1], _schwarzschild(r, v), rtol=1e-12, atol=0)


def test_a_moving_sun_is_what_the_one_body_term_leaves_out():
    """Boost everything by the Sun's ~15 m/s barycentric velocity: the
    heliocentric EIH term changes. That velocity dependence is exactly what a
    heliocentric one-body model cannot represent, at ~v_sun/v_mercury ~ 1e-4
    to 1e-3 of the GR term -- the size of the residual it explains."""
    r, v = _mercury_like_states()
    zero = np.zeros_like(r)
    mu = np.array([GM_SUN, 0.0])
    boost = np.array([0.0, 15.0, 0.0])
    rest = _eih_1pn(np.array([zero, r]), np.array([zero, v]), mu)
    moving = _eih_1pn(np.array([zero, r]), np.array([zero + boost, v + boost]), mu)
    rel = np.linalg.norm((moving[1] - moving[0]) - (rest[1] - rest[0]), axis=1)
    rel /= np.linalg.norm(rest[1], axis=1)
    assert 1e-5 < np.median(rel) < 1e-2
