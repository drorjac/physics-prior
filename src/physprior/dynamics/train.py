"""Training a stepper: one-step loss or an unrolled (multi-step) loss.

Both objectives use the same number of gradient steps, the same batch of
windows and the same optimiser, so they differ only in what the loss sees.
The unrolled loss costs `horizon` times more forward work per step; the
seconds are recorded so that cost is reported next to the benefit.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from physprior.units import require

from .steppers import DTYPE


@dataclass(frozen=True)
class TrainOptions:
    steps: int = 2000
    batch: int = 128
    lr: float = 3e-3
    loss: str = "onestep"  # "onestep" | "rollout"
    horizon: int = 8  # unroll length of the rollout loss
    clip: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


def windows(traj: np.ndarray, length: int) -> np.ndarray:
    """All windows of `length` consecutive states: (M, length, ...)."""
    n_t = traj.shape[1]
    require(n_t >= length, "trajectory shorter than the training window")
    idx = np.arange(n_t - length + 1)[:, None] + np.arange(length)[None, :]
    w = traj[:, idx]  # (n_traj, M, length, ...)
    return w.reshape(-1, length, *traj.shape[2:])


def train(
    model: torch.nn.Module,
    traj: np.ndarray,
    scale,
    opts: TrainOptions,
    seed: int,
) -> dict:
    """Fit `model` to trajectories (n_traj, T+1, ...). Returns the loss
    history and the wall time."""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    K = opts.horizon if opts.loss == "rollout" else 1
    W = torch.as_tensor(windows(traj, K + 1), dtype=DTYPE)
    s = torch.as_tensor(np.asarray(scale, float), dtype=DTYPE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=opts.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, opts.steps)
    hist: list[float] = []
    t0 = time.time()
    for it in range(opts.steps):
        b = W[gen.integers(0, len(W), min(opts.batch, len(W)))]
        u = b[:, 0]
        loss = torch.zeros((), dtype=DTYPE)
        for k in range(1, K + 1):
            u = model(u)
            loss = loss + torch.mean(((u - b[:, k]) / s) ** 2)
        loss = loss / K
        opt.zero_grad()
        if not torch.isfinite(loss):
            hist.append(float("nan"))
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, opts.clip)
        opt.step()
        sched.step()
        if it % 50 == 0 or it == opts.steps - 1:
            hist.append(loss.item())
    return {"loss": hist, "seconds": time.time() - t0, "final_loss": hist[-1]}
