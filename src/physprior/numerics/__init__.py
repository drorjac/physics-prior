"""Numerical machinery, and the discipline that goes with it.

Integrators and finite-difference stencils, each carrying the means to check
itself: conserved quantities for the integrators, convergence order for the
stencils. A result that has not stopped moving with step size is not a result.
"""

from __future__ import annotations

from .integrators import Trajectory, integrate
from .stencils import derivative, richardson

__all__ = ["Trajectory", "derivative", "integrate", "richardson"]
