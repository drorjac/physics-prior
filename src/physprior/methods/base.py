"""What every method returns, so the tracks can be compared at all."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Fit:
    """One trained model on one dataset."""

    name: str  # "nn" | "pinn" | "sr" | "oracle"
    predict: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    params: dict[str, float] = field(default_factory=dict)  # recovered physics
    param_sigma: dict[str, float] = field(default_factory=dict)
    expression: str | None = None  # closed form, if the method has one
    n_free: int = 0  # free parameters, for chi^2/dof
    seconds: float = 0.0
    # Per-epoch training record: {"epoch": [...], "loss": [...], "pred": [...],
    # plus one entry per trainable physical constant}. None when not recorded.
    history: dict[str, list] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def interpretable(self) -> bool:
        """Can a physicist read a law or a constant off this model?"""
        return self.expression is not None or bool(self.params)


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# The project's seed discipline, carried over from LR-DSR: choose
# hyperparameters on the tuning seeds, report only on the reporting seeds.
TUNE_SEEDS = (3, 7, 19)
REPORT_SEEDS = (11, 23, 42)
