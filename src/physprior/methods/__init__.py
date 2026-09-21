"""The model families being compared.

oracle    the published law with published constants -- the ceiling
physics   the published law with its constants fitted (`pinn.fit_physics`)
pinn      the law plus a neural correction, or an ODE residual
sr        symbolic regression (`symbolic.fit_sr`)
nn        a plain MLP that knows no physics (`neural.train_mlp`)
"""

from __future__ import annotations

from .base import REPORT_SEEDS, TUNE_SEEDS, Fit

__all__ = ["REPORT_SEEDS", "TUNE_SEEDS", "Fit"]
