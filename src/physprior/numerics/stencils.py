"""Finite-difference derivatives, and Richardson extrapolation.

Both exist in one place because both are used where the quantity of interest
is smaller than a careless discretisation error:

  * Mercury's relativistic term is 8e-8 of its acceleration, and a 4th-order
    stencil at a 3-hour step leaves a truncation error a few per cent of that;
  * hydrogen's QED shift is 10.8 ppm, and plain second-order finite
    differences give the levels to 300 ppm.

`derivative` therefore offers explicit orders so the order can be swept, and
`richardson` cancels the leading error term from two resolutions.
"""

from __future__ import annotations

import numpy as np

from physprior.exceptions import ConvergenceError

#: Central-difference coefficients, by accuracy order.
_STENCILS: dict[int, tuple[tuple[int, float], ...]] = {
    2: ((-1, -0.5), (1, 0.5)),
    4: ((-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12)),
    6: (
        (-3, -1 / 60),
        (-2, 9 / 60),
        (-1, -45 / 60),
        (1, 45 / 60),
        (2, -9 / 60),
        (3, 1 / 60),
    ),
}


def derivative(y: np.ndarray, h: float, order: int = 6, axis: int = 0) -> np.ndarray:
    """Central difference of `y` along `axis`, on a uniform grid of step `h`.

    Edges where the stencil does not fit are filled with NaN rather than a
    lower-order formula: a silently less accurate edge is how a truncation
    error becomes a physical claim.
    """
    if order not in _STENCILS:
        raise ConvergenceError(
            f"unsupported stencil order {order}; have {sorted(_STENCILS)}"
        )
    y = np.asarray(y, float)
    y = np.moveaxis(y, axis, 0)
    half = order // 2
    if y.shape[0] <= order:
        raise ConvergenceError(
            f"need more than {order} samples for an order-{order} stencil, "
            f"got {y.shape[0]}"
        )
    out = np.full_like(y, np.nan)
    core = slice(half, y.shape[0] - half)
    acc = np.zeros_like(y[core])
    for offset, weight in _STENCILS[order]:
        lo = half + offset
        acc = acc + weight * y[lo : lo + (y.shape[0] - order)]
    out[core] = acc / h
    return np.moveaxis(out, 0, axis)


def richardson(
    coarse: np.ndarray, fine: np.ndarray, order: int = 2, refinement: int = 2
) -> np.ndarray:
    """Cancel the leading O(h^order) error from two resolutions.

    `fine` is computed at step h/`refinement`. For the usual second-order,
    twice-refined case this is `fine + (fine - coarse)/3`.
    """
    factor = refinement**order - 1.0
    return np.asarray(fine) + (np.asarray(fine) - np.asarray(coarse)) / factor
