"""The network, its derivatives, and the two training loops.

The network maps s = 2 t / T - 1 to a scalar. Its first and second
derivatives in s are propagated through the layers in forward mode, in the
same pass as the value (as in `lorenz/pinn.py`), so the ODE residual costs
one pass and no double backward.

Trial A (`train_forward`): the initial condition is built in,
u = u0 + v0 t + t^2 N(s), and the loss is the residual alone,

    r_i = (u'' + F(u, u')) / k,     L = mean_i w_i r_i^2,

with w from a `ResidualRule`. Trial B (`train_inverse`): u = mean + sd N(s),
k = k0 exp(theta) with theta trainable, and

    L = mean((u - y) / sd)^2 + lambda * mean(((u'' + F) / (k0 sd))^2),

with lambda from a `BalanceRule`. Everything else (width, depth, Adam with a
cosine schedule, step budget, stratified collocation resampled every step) is
fixed in `TrainConfig` and shared by every rule.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .rules import BalanceRule, InvalidRule, ResidualRule
from .tasks import Inverse, Task, as_tensor

DTYPE = torch.float64
ERR_CLIP = 10.0  # relative errors above this count as this


@dataclass(frozen=True)
class TrainConfig:
    width: int = 32
    depth: int = 3
    steps: int = 3000
    lr: float = 1e-2
    n_col: int = 128
    grad_every: int = 10  # trial B: gradient-norm feature refresh

    def quick(self) -> TrainConfig:
        return TrainConfig(self.width, self.depth, 300, self.lr, 64, self.grad_every)


class Net(nn.Module):
    """tanh MLP R -> R returning (N, dN/ds, d2N/ds2)."""

    def __init__(self, width: int, depth: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        sizes = [1] + [width] * depth
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, b in itertools.pairwise(sizes):
            std = math.sqrt(2.0 / (a + b))  # Xavier, for tanh
            self.W.append(
                nn.Parameter(torch.randn(b, a, generator=g, dtype=DTYPE) * std)
            )
            self.b.append(nn.Parameter(torch.zeros(b, dtype=DTYPE)))
        std = math.sqrt(2.0 / (width + 1))
        self.Wo = nn.Parameter(torch.randn(1, width, generator=g, dtype=DTYPE) * std)
        self.bo = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = s
        da = torch.ones_like(s)
        dda = torch.zeros_like(s)
        for W, b in zip(self.W, self.b, strict=True):
            z = a @ W.T + b
            dz = da @ W.T
            ddz = dda @ W.T
            a = torch.tanh(z)
            s1 = 1 - a * a
            da, dda = s1 * dz, s1 * ddz - 2 * a * s1 * dz * dz
        return a @ self.Wo.T + self.bo, da @ self.Wo.T, dda @ self.Wo.T


def _collocation(rng: np.random.Generator, n: int) -> np.ndarray:
    """Stratified in [0, 1], sorted, so cumulative sums run forward in time."""
    return (np.arange(n) + rng.random(n)) / n


def _rel_error(pred: np.ndarray, ref: np.ndarray) -> float:
    e = float(np.sqrt(np.mean((pred - ref) ** 2)) / np.std(ref))
    return min(e, ERR_CLIP) if np.isfinite(e) else ERR_CLIP


# ---------------------------------------------------------------------------
# trial A
# ---------------------------------------------------------------------------


def _forward_u(net: Net, task: Task, t: torch.Tensor):
    T = task.T
    N, dN, ddN = net(2 * t / T - 1)
    d = 2.0 / T
    u = task.u0 + task.v0 * t + t * t * N
    du = task.v0 + 2 * t * N + t * t * dN * d
    ddu = 2 * N + 4 * t * dN * d + t * t * ddN * d * d
    return u, du, ddu


def train_forward(
    task: Task, rule: ResidualRule, seed: int, cfg: TrainConfig = TrainConfig()
) -> dict:
    """Train one PINN on one forward task. Returns the error against the
    reference, and `valid=False` if the rule produced non-finite weights."""
    torch.manual_seed(seed)
    net = Net(cfg.width, cfg.depth, seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    rng = np.random.default_rng(seed)
    k = task.k
    valid = True
    for step in range(cfg.steps):
        x = _collocation(rng, cfg.n_col)
        t = as_tensor(x * task.T)[:, None]
        u, du, ddu = _forward_u(net, task, t)
        r = ((ddu + task.force(u, du, lib=torch)) / k).squeeze(1)
        r2 = r * r
        try:
            w = rule.weights(x, r2.detach().numpy(), step / cfg.steps)
        except InvalidRule:
            valid = False
            break
        loss = (as_tensor(w) * r2).mean()
        if not torch.isfinite(loss):
            valid = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    t_ref, u_ref = task.solution()
    with torch.no_grad():
        u, _, _ = _forward_u(net, task, as_tensor(t_ref)[:, None])
    err = _rel_error(u.squeeze(1).numpy(), u_ref) if valid else ERR_CLIP
    return {"task": task.name, "seed": seed, "err": err, "valid": valid}


# ---------------------------------------------------------------------------
# trial B
# ---------------------------------------------------------------------------


def _grad_norm(loss: torch.Tensor, params: Sequence[torch.Tensor]) -> float:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float((x * x).sum()) for x in g if x is not None))


def train_inverse(
    inv: Inverse, rule: BalanceRule, seed: int, cfg: TrainConfig = TrainConfig()
) -> dict:
    """Train one PINN on one inverse task; return the errors in k and in u."""
    task = inv.task
    torch.manual_seed(seed)
    net = Net(cfg.width, cfg.depth, seed)
    t_obs, y_obs = inv.observations(seed)
    mu, sd = float(np.mean(y_obs)), float(np.std(y_obs))
    k0 = inv.k_init * task.k
    theta = nn.Parameter(torch.zeros((), dtype=DTYPE))
    params = list(net.parameters())
    opt = torch.optim.Adam([*params, theta], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    rng = np.random.default_rng(seed)
    T = task.T
    d = 2.0 / T
    so = as_tensor(2 * t_obs / T - 1)[:, None]
    yo = as_tensor(y_obs)[:, None]
    gamma = 0.0
    valid = True
    for step in range(cfg.steps):
        n_obs, _, _ = net(so)
        ld = (((mu + sd * n_obs) - yo) / sd).pow(2).mean()
        x = _collocation(rng, cfg.n_col)
        N, dN, ddN = net(as_tensor(2 * x - 1)[:, None])
        u = mu + sd * N
        du = sd * dN * d
        ddu = sd * ddN * d * d
        k = k0 * torch.exp(theta)
        lp = ((ddu + task.force(u, du, k=k, lib=torch)) / (k0 * sd)).pow(2).mean()
        rho = math.log10(max(ld.item(), 1e-300) / max(lp.item(), 1e-300))
        if rule.needs_gradients and step % cfg.grad_every == 0:
            gd, gp = _grad_norm(ld, params), _grad_norm(lp, params)
            gamma = math.log10(max(gd, 1e-300) / max(gp, 1e-300))
        try:
            lam = rule.weight(step / cfg.steps, rho, gamma)
        except InvalidRule:
            valid = False
            break
        loss = ld + lam * lp
        if not torch.isfinite(loss):
            valid = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    t_ref, u_ref = task.solution()
    with torch.no_grad():
        N, _, _ = net(as_tensor(2 * t_ref / T - 1)[:, None])
    k_hat = k0 * math.exp(theta.item()) if valid else float("nan")
    err_k = abs(k_hat / task.k - 1) if np.isfinite(k_hat) else ERR_CLIP
    err_u = _rel_error((mu + sd * N).squeeze(1).numpy(), u_ref) if valid else ERR_CLIP
    return {
        "task": inv.name,
        "seed": seed,
        "err_k": min(err_k, ERR_CLIP),
        "err_u": err_u,
        "k_hat": k_hat,
        "valid": valid,
    }
