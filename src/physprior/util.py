"""Small shared helpers."""

from __future__ import annotations

import numpy as np


def first_column(x) -> np.ndarray:
    """The first input variable, whatever shape the caller passed.

    `np.atleast_2d` on a 1-D array of length N gives shape (1, N), so the
    obvious `np.atleast_2d(x)[:, 0]` silently returns ONE element instead of
    N. Every law in this project takes its input through here instead; a test
    caught the difference and it would otherwise have been a fit that
    converged to a bound for no visible reason.
    """
    a = np.asarray(x, float)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a[:, 0]
    raise ValueError(f"expected 1-D or 2-D input, got shape {a.shape}")


def as_columns(x) -> np.ndarray:
    """Inputs as (N, d), accepting (N,) as a single column."""
    a = np.asarray(x, float)
    return a.reshape(-1, 1) if a.ndim == 1 else a
