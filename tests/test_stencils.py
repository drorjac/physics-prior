"""Finite differences: the accuracy order must be what it claims.

This is not pedantry. A 4th-order stencil at a plausible step size once
produced a 56-sigma "violation of general relativity" in this project, purely
from truncation error.
"""

from __future__ import annotations

import numpy as np
import pytest

from physprior.exceptions import ConvergenceError
from physprior.numerics.stencils import derivative, richardson

# Resolutions chosen per order so the TRUNCATION error still dominates. An
# order-6 stencil on 201 points is already at 3e-13 -- the floating-point
# roundoff floor -- and refining further measures noise, not convergence. The
# first version of this test used (201, 401) for every order and "found" that
# the order-6 stencil converges at 1.14.
_GRIDS = {2: (101, 201), 4: (21, 41), 6: (21, 41)}


@pytest.mark.parametrize("order,expected", [(2, 2.0), (4, 4.0), (6, 6.0)])
def test_convergence_order(order, expected):
    """Halving h must cut the error by 2**order."""
    errs = []
    for n in _GRIDS[order]:
        x = np.linspace(0.0, 1.0, n)
        d = derivative(np.sin(3 * x), x[1] - x[0], order=order)
        m = np.isfinite(d)
        errs.append(np.abs(d[m] - 3 * np.cos(3 * x)[m]).max())
    # guard the guard: if the finer run has hit roundoff, the ratio below is
    # meaningless and the test would be measuring nothing.
    assert errs[1] > 1e-12, (
        f"order-{order} at n={_GRIDS[order][1]} is at the roundoff floor "
        f"({errs[1]:.1e}); this test would measure noise"
    )
    measured = np.log2(errs[0] / errs[1])
    assert abs(measured - expected) < 0.6, (
        f"order-{order} stencil converges at {measured:.2f}, not {expected}"
    )


def test_edges_are_nan_not_silently_lower_order():
    d = derivative(np.arange(20.0), 1.0, order=6)
    assert np.isnan(d[:3]).all() and np.isnan(d[-3:]).all()
    assert np.allclose(d[3:-3], 1.0)


def test_unsupported_order_is_an_error():
    with pytest.raises(ConvergenceError):
        derivative(np.arange(20.0), 1.0, order=3)


def test_too_few_samples_is_an_error():
    with pytest.raises(ConvergenceError):
        derivative(np.arange(4.0), 1.0, order=6)


def test_richardson_cancels_the_leading_term():
    """A second-order quantity extrapolated from two grids beats both."""

    def estimate(h):
        return np.pi + 0.5 * h**2 + 0.01 * h**3

    coarse, fine = estimate(0.1), estimate(0.05)
    combined = richardson(coarse, fine, order=2, refinement=2)
    assert abs(combined - np.pi) < 0.1 * abs(fine - np.pi)
