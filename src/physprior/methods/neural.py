"""The black-box baseline: a plain MLP that knows nothing about physics.

This arm is deliberately given a fair chance. Its width, depth and weight
decay are chosen by a grid search on the TUNING seeds and then frozen, so a
poor showing is a property of the data budget and not of a handicapped
baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import Fit, set_seed

DTYPE = torch.float64


def mlp(d_in: int, width: int, depth: int, d_out: int = 1) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d_in, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, d_out)]
    return nn.Sequential(*layers).to(DTYPE)


@dataclass
class Standardiser:
    """Physics data spans decades. Networks do not train on raw units."""

    mu_x: np.ndarray
    sd_x: np.ndarray
    mu_y: float
    sd_y: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> Standardiser:
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[0] != len(y):
            x = x.T
        sd_x = x.std(axis=0)
        sd_x[sd_x == 0] = 1.0
        sd_y = float(np.std(y)) or 1.0
        return cls(x.mean(axis=0), sd_x, float(np.mean(y)), sd_y)

    def x(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != len(self.mu_x):
            x = x.T
        return (x - self.mu_x) / self.sd_x

    def y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, float) - self.mu_y) / self.sd_y

    def y_inv(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) * self.sd_y + self.mu_y


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int = 32,
    depth: int = 3,
    weight_decay: float = 1e-4,
    epochs: int = 4000,
    lr: float = 5e-3,
    seed: int = 0,
    record_every: int = 0,
    record_grid: np.ndarray | None = None,
) -> Fit:
    """`record_every > 0` keeps a training history: the loss at each recorded
    epoch and, if `record_grid` is given, the model's prediction on that grid.
    That is what makes the training watchable rather than merely reported."""
    t0 = time.time()
    set_seed(seed)
    std = Standardiser.fit(x, y)
    xs = torch.tensor(std.x(x), dtype=DTYPE)
    ys = torch.tensor(std.y(y), dtype=DTYPE).reshape(-1, 1)

    net = mlp(xs.shape[1], width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: dict[str, list] = {"epoch": [], "loss": [], "pred": []}
    grid_t = None
    if record_every and record_grid is not None:
        g = np.asarray(record_grid, float)
        grid_t = torch.tensor(
            std.x(g.reshape(-1, 1) if g.ndim == 1 else g), dtype=DTYPE
        )

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(xs) - ys) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["loss"].append(float(loss.detach()))
            if grid_t is not None:
                with torch.no_grad():
                    history["pred"].append(std.y_inv(net(grid_t).numpy().ravel()))

    def predict(xq: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = net(torch.tensor(std.x(xq), dtype=DTYPE)).numpy().ravel()
        return std.y_inv(z)

    n_free = sum(p.numel() for p in net.parameters())
    return Fit(
        name="nn",
        predict=predict,
        n_free=n_free,
        seconds=time.time() - t0,
        history=history if record_every else None,
        extra={
            "width": width,
            "depth": depth,
            "weight_decay": weight_decay,
            "final_train_mse_std_units": float(loss.item()),
        },
    )


# Grid searched on the tuning seeds only. Small, because the datasets are small.
NN_GRID: list[dict[str, Any]] = [
    {"width": 16, "depth": 2, "weight_decay": 1e-2},
    {"width": 16, "depth": 2, "weight_decay": 1e-4},
    {"width": 32, "depth": 3, "weight_decay": 1e-3},
    {"width": 32, "depth": 3, "weight_decay": 1e-5},
    {"width": 64, "depth": 4, "weight_decay": 1e-4},
]


def tune_mlp(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    tune_seeds,
    epochs: int = 4000,
) -> dict:
    """Pick MLP hyperparameters on held-out data, averaged over tuning seeds."""
    from physprior.benchmark.metrics import rmse

    best: dict[str, Any] = NN_GRID[0]
    best_score = np.inf
    for cfg in NN_GRID:
        scores = []
        for s in tune_seeds:
            f = train_mlp(x, y, seed=s, epochs=epochs, **cfg)
            scores.append(rmse(y_val, f.predict(x_val)))
        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = cfg, score
    return dict(best, tuned_val_rmse=best_score)
