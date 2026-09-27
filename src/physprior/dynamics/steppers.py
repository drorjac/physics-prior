"""Learned steppers: maps u_n -> u_{n+1} with different amounts of structure.

All steppers see states in physical units and standardise internally with
statistics of the training set, so the comparison is between architectures
and not between input scalings.

ODE steppers
------------
direct        u' = NN(u)                     black box, must learn the identity
residual      u' = u + dt NN(u)              Euler-like increment
node          u' = RK4_dt[NN](u)             neural vector field, trained through RK4
hnn           u' = RK4_dt[J grad H_NN](u)    Hamiltonian field (Greydanus et al. 2019)
hnn_leapfrog  leapfrog on H = T_NN(p) + V_NN(q), symplectic by construction
closure       u' = RK4_dt[f_known + NN](u)   known physics plus a learned remainder
oracle        u' = RK4_dt[f_true](u)         the true field, one RK4 step: a reference
known_only    u' = RK4_dt[f_known](u)        the incomplete physics without closure

PDE steppers (periodic 1-D grid)
--------------------------------
conv          u' = u + dt CNN(u)             kernel-5 conv then 1x1 convs: local,
                                             translation-equivariant
dense         u' = u + dt MLP(u)             MLP over the whole grid
pde_closure   u' = u + dt (f_known_h(u) + CNN(u))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from physprior.units import require_not_none

from .systems import OdeSystem, torch_ops

DTYPE = torch.float64


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, float), dtype=DTYPE)


def mlp(d_in: int, d_out: int, width: int = 64, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(d_in, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, d_out))
    return nn.Sequential(*layers).to(DTYPE)


def rk4_step(f: Callable[[torch.Tensor], torch.Tensor], u: torch.Tensor, h: float):
    k1 = f(u)
    k2 = f(u + 0.5 * h * k1)
    k3 = f(u + 0.5 * h * k2)
    k4 = f(u + h * k3)
    return u + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class Stepper(nn.Module):
    """Base: holds the standardisation and the step size."""

    #: True when a step needs autograd (the Hamiltonian arms).
    needs_grad = False
    mu: torch.Tensor
    sd: torch.Tensor
    dsd: torch.Tensor

    def __init__(self, dt: float, mean, std, dstd):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("mu", _t(mean))
        self.register_buffer("sd", _t(std))
        self.register_buffer("dsd", _t(dstd))

    def z(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.mu) / self.sd


class Direct(Stepper):
    def __init__(self, dim, dt, mean, std, dstd, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(dim, dim, width, depth)

    def forward(self, u):
        return self.mu + self.sd * self.net(self.z(u))


class Residual(Stepper):
    def __init__(self, dim, dt, mean, std, dstd, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(dim, dim, width, depth)

    def forward(self, u):
        return u + self.dt * self.dsd * self.net(self.z(u))


class NeuralODE(Stepper):
    def __init__(self, dim, dt, mean, std, dstd, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(dim, dim, width, depth)

    def field(self, u):
        return self.dsd * self.net(self.z(u))

    def forward(self, u):
        return rk4_step(self.field, u, self.dt)


class Closure(Stepper):
    """Known field plus a learned remainder, integrated together with RK4.
    The remainder is scaled by the spread of what the known field misses, so
    the network starts near 'physics only' rather than near noise."""

    def __init__(self, dim, dt, mean, std, dstd, known, rscale, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(dim, dim, width, depth)
        self.known = known
        self.ops = torch_ops()
        self.register_buffer("rsd", _t(rscale))

    def field(self, u):
        return self.known(u, self.ops) + self.rsd * self.net(self.z(u))

    def forward(self, u):
        return rk4_step(self.field, u, self.dt)


class Fixed(Stepper):
    """A given vector field and RK4: the oracle, or the known part alone."""

    def __init__(self, dim, dt, mean, std, dstd, f):
        super().__init__(dt, mean, std, dstd)
        self.f = f
        self.ops = torch_ops()

    def forward(self, u):
        return rk4_step(lambda x: self.f(x, self.ops), u, self.dt)


class HNN(Stepper):
    """Greydanus, Dzamba & Yosinski (2019): a scalar network H(q, p) whose
    symplectic gradient is the vector field. Integrated here with RK4, which
    is not symplectic, so this arm isolates the value of a Hamiltonian field
    from the value of a symplectic integrator."""

    needs_grad = True

    def __init__(self, dim, dt, mean, std, dstd, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(dim, 1, width, depth)
        self.nq = dim // 2
        # H has units of (q-scale * p-rate); one scalar is enough.
        self.register_buffer(
            "hs", _t(float(np.mean(np.asarray(std) * np.asarray(dstd))))
        )

    def H(self, u):
        return self.hs * self.net(self.z(u)).squeeze(-1)

    def field(self, u):
        if not u.requires_grad:
            u = u.requires_grad_(True)
        g = torch.autograd.grad(self.H(u).sum(), u, create_graph=True)[0]
        nq = self.nq
        return torch.cat([g[..., nq:], -g[..., :nq]], dim=-1)

    def forward(self, u):
        return rk4_step(self.field, u, self.dt)


class HNNLeapfrog(Stepper):
    """Separable H = T(p) + V(q) with a leapfrog (Stormer-Verlet) update, as
    in the symplectic recurrent networks of Chen et al. (2020). The update is
    an exactly symplectic map for whatever T and V the networks represent."""

    needs_grad = True

    def __init__(self, dim, dt, mean, std, dstd, width=64, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.nq = dim // 2
        self.V = mlp(self.nq, 1, width, depth)
        self.T = mlp(self.nq, 1, width, depth)
        self.register_buffer(
            "hs", _t(float(np.mean(np.asarray(std) * np.asarray(dstd))))
        )

    def _grad(self, net, x, mu, sd):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        e = self.hs * net((x - mu) / sd).sum()
        return torch.autograd.grad(e, x, create_graph=True)[0]

    def dV(self, q):
        nq = self.nq
        return self._grad(self.V, q, self.mu[:nq], self.sd[:nq])

    def dT(self, p):
        nq = self.nq
        return self._grad(self.T, p, self.mu[nq:], self.sd[nq:])

    def H(self, u):
        nq = self.nq
        q, p = u[..., :nq], u[..., nq:]
        v = self.V((q - self.mu[:nq]) / self.sd[:nq]).squeeze(-1)
        t = self.T((p - self.mu[nq:]) / self.sd[nq:]).squeeze(-1)
        return self.hs * (v + t)

    def forward(self, u):
        nq, h = self.nq, self.dt
        q, p = u[..., :nq], u[..., nq:]
        p = p - 0.5 * h * self.dV(q)
        q = q + h * self.dT(p)
        p = p - 0.5 * h * self.dV(q)
        return torch.cat([q, p], dim=-1)


ODE_ARMS = ("direct", "residual", "node", "hnn", "hnn_leapfrog", "closure")
REFERENCE_ARMS = ("oracle", "known_only")


def arms_for(sys: OdeSystem) -> list[str]:
    arms = ["direct", "residual", "node"]
    if sys.hamiltonian:
        arms += ["hnn", "hnn_leapfrog"]
    if sys.known is not None:
        arms.append("closure")
    return arms


def build_ode(arm: str, sys: OdeSystem, data, width: int = 64, depth: int = 2):
    a = (sys.dim, sys.dt, data.mean, data.std, data.dstd)
    if arm == "direct":
        return Direct(*a, width=width, depth=depth)
    if arm == "residual":
        return Residual(*a, width=width, depth=depth)
    if arm == "node":
        return NeuralODE(*a, width=width, depth=depth)
    if arm == "hnn":
        return HNN(*a, width=width, depth=depth)
    if arm == "hnn_leapfrog":
        return HNNLeapfrog(*a, width=width, depth=depth)
    if arm == "closure":
        return Closure(
            *a, sys.known, closure_scale(sys, data), width=width, depth=depth
        )
    if arm == "oracle":
        return Fixed(*a, sys.f)
    if arm == "known_only":
        return Fixed(*a, sys.known)
    raise KeyError(arm)


def closure_scale(sys: OdeSystem, data) -> np.ndarray:
    """Spread of (finite-difference rate - known field) on the training set,
    floored so components the known physics gets exactly right still train."""
    from .systems import NP

    u = data.train[:, :-1].reshape(-1, sys.dim)
    rate = ((data.train[:, 1:] - data.train[:, :-1]) / sys.dt).reshape(-1, sys.dim)
    known = require_not_none(sys.known, f"{sys.name} has no known part")
    miss = rate - known(u, NP)
    s = miss.std(axis=0)
    return np.maximum(s, 1e-3 * data.dstd)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# PDE steppers


class ConvNet(nn.Module):
    """Receptive field = the first kernel; later layers are pointwise."""

    def __init__(self, channels: int = 32, kernel: int = 5, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(
                1, channels, kernel, padding=kernel // 2, padding_mode="circular"
            ),
            nn.Tanh(),
        ]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(channels, channels, 1), nn.Tanh()]
        layers.append(nn.Conv1d(channels, 1, 1))
        self.net = nn.Sequential(*layers).to(DTYPE)

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


class PdeStepper(nn.Module):
    needs_grad = False

    def __init__(self, dt: float, mean: float, std: float, dstd: float):
        super().__init__()
        self.dt = float(dt)
        self.mu, self.sd, self.dsd = float(mean), float(std), float(dstd)

    def z(self, u):
        return (u - self.mu) / self.sd


class ConvStepper(PdeStepper):
    def __init__(self, n, dt, mean, std, dstd, channels=32, kernel=5, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = ConvNet(channels, kernel, depth)

    def forward(self, u):
        return u + self.dt * self.dsd * self.net(self.z(u))


class DenseStepper(PdeStepper):
    def __init__(self, n, dt, mean, std, dstd, width=256, depth=2):
        super().__init__(dt, mean, std, dstd)
        self.net = mlp(n, n, width, depth)

    def forward(self, u):
        return u + self.dt * self.dsd * self.net(self.z(u))


class PdeClosure(PdeStepper):
    """Known discretised term plus a local learned remainder, one Euler step."""

    def __init__(
        self, n, dt, mean, std, dstd, known, rscale, channels=32, kernel=5, depth=2
    ):
        super().__init__(dt, mean, std, dstd)
        self.net = ConvNet(channels, kernel, depth)
        self.known = known
        self.rsd = float(rscale)

    def forward(self, u):
        return u + self.dt * (self.known(u) + self.rsd * self.net(self.z(u)))


class PdeFixed(PdeStepper):
    def __init__(self, dt, step):
        super().__init__(dt, 0.0, 1.0, 1.0)
        self.step = step

    def forward(self, u):
        return self.step(u)
