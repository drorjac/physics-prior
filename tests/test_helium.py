"""quantum/helium: the complete-law comparison and the defect diagnostic.

These run on synthetic He-like terms with known quantum defects, so they
need no download and run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from physprior.benchmark.protocol import Problem
from physprior.constants import HE_I_IONISATION_CM, RYDBERG_HE_CM
from physprior.methods.base import Fit
from physprior.problems.quantum import helium as H

DEFECTS = {(0, 0): 0.14, (0, 1): 0.30, (1, 0): -0.012, (1, 1): 0.068, (2, 0): 0.002}


def _terms(n_min: int = 4, n_max: int = 20):
    """Every series over the same n range, as in the real test split."""
    rows = [
        (n, ell, s, HE_I_IONISATION_CM - RYDBERG_HE_CM / (n - d) ** 2)
        for (ell, s), d in DEFECTS.items()
        for n in range(n_min, n_max + 1)
    ]
    a = np.array(rows, float)
    return a[:, :3], a[:, 3]


def _problem():
    x, y = _terms()
    return Problem(
        track=H.TRACK,
        x=x,
        y=y,
        law_np=H.law_np,
        law_t=H.law_t,
        params=[],
        theta_published={},
    )


def test_rydberg_ritz_recovers_the_defects_and_the_limit():
    x, y = _terms()
    fit = H.fit_rydberg_ritz(x, y)
    for (ell, s), d in DEFECTS.items():
        got = fit.extra["defects"][f"{H.L_LETTERS[ell]}{2 * s + 1}"]["d0"]
        assert got == pytest.approx(d, abs=1e-6)
    assert fit.params["L"] == pytest.approx(HE_I_IONISATION_CM, abs=1e-4)
    np.testing.assert_allclose(fit.predict(x), y, rtol=1e-10)


def test_the_implied_defect_of_the_data_is_the_defect():
    x, y = _terms()
    d = H.implied_defect(x, y)
    for (ell, s), want in DEFECTS.items():
        sel = (x[:, 1] == ell) & (x[:, 2] == s)
        np.testing.assert_allclose(d[sel], want, atol=1e-9)


def test_a_law_blind_to_l_shows_no_l_structure():
    """Over matched n ranges, the hydrogenic law gives every series the same
    implied defect, whatever its constants."""
    prob = _problem()
    hydrogenic = Fit(
        name="physics",
        predict=lambda xq: H.law_np(xq, L=HE_I_IONISATION_CM + 50.0, R=1.1e5),
    )
    table = H.defect_table(prob, {"physics": hydrogenic}, np.arange(len(prob)))
    per_series = table.groupby("series")["physics"].first()
    assert per_series.max() - per_series.min() < 1e-9


def test_the_extrapolation_split_is_exactly_n_le_10():
    x, y = _terms(n_min=2)
    prob = Problem(
        track=H.TRACK,
        x=x,
        y=y,
        law_np=H.law_np,
        law_t=H.law_t,
        params=[],
        theta_published={},
    )
    itr, ite = H.extrapolation_split(prob)
    assert prob.x[itr, 0].max() == 10 and prob.x[ite, 0].min() == 11
    assert len(itr) + len(ite) == len(prob)
