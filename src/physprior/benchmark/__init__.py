"""The evaluation protocol, identical for every problem.

Six questions, asked the same way every time: interpolation, data efficiency,
noise, extrapolation, parameter recovery, law recovery.
"""

from __future__ import annotations

from .metrics import chi2_reduced, nrmse, rmse
from .protocol import ARMS, Problem

__all__ = ["ARMS", "Problem", "chi2_reduced", "nrmse", "rmse"]
