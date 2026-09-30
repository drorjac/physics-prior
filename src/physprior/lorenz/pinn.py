"""The network arms: a black box and a physics-informed network (PINN).

Both map time to state, u(t) = mu + sd * N(s), with s = 2 t / T - 1 in
[-1, 1] and mu, sd the mean and sd of the observations. The black box is
trained on the observations alone:

    L_data = mean_i,k ((u_k(t_i) - y_ik) / sd_k)^2

The PINN adds the residual of the law at collocation times t_j, with the three
constants trainable (log-parametrised, so they stay positive):

    r_j = du/dt(t_j) - f(u(t_j); sigma, rho, beta)
    L_phys = mean_j,k (r_jk / (sd_k * 2/T))^2
    L = L_data + w(step) * L_phys

The scale 2/T makes the residual dimensionless in the same units as the
network's own derivative dN/ds.

The training recipe, every piece of which is a flag in `PinnConfig` so the
optimisation ladder in `study.py` can switch them on one at a time:

    warmup, ramp    fit the data alone first, then raise w from 0 to w_phys.
                    Without it the residual term, which is ~1e5 times the data
                    term at initialisation, wins: the network settles on a
                    trajectory that satisfies the law with the wrong constants
                    and ignores the data.
    schedule        cosine decay of the Adam learning rate
    resample        new collocation times every step (stratified in time)
    lbfgs           a quasi-Newton finish on a fixed collocation grid
    fourier         sin/cos(k pi s / 2) input features, k = 1..fourier
    balance         "gradnorm": w rescaled by the ratio of gradient norms
                    (Wang, Teng & Perdikaris 2021)
    causal_eps      residual weights that open in time (Wang, Sankaran &
                    Perdikaris 2024)
    derivative      how du/dt is computed: "forward" (by hand, one pass),
                    "jvp" (torch.func), "autograd" (reverse mode, one pass per
                    output). They agree to rounding; they differ in cost.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from physprior.methods.neural import DTYPE
from physprior.units import require

from .system import THETA_INIT, THETA_NAMES, Observations


class LorenzNet(nn.Module):
    """tanh MLP from s in [-1, 1] to three standardised state components."""

    freq: torch.Tensor

    def __init__(self, width: int = 64, depth: int = 4, fourier: int = 0):
        super().__init__()
        self.fourier = fourier
        d_in = 1 + 2 * fourier
        freq = torch.arange(1, fourier + 1, dtype=DTYPE) * (math.pi / 2)
        self.register_buffer("freq", freq)
        sizes = [d_in] + [width] * depth
        self.hidden: list[nn.Linear] = nn.ModuleList(  # type: ignore[assignment]
            nn.Linear(a, b, dtype=DTYPE) for a, b in itertools.pairwise(sizes)
        )
        self.out = nn.Linear(width, 3, dtype=DTYPE)

    def _features(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fourier:
            return s, torch.ones_like(s)
        a = s * self.freq
        h = torch.cat([s, torch.sin(a), torch.cos(a)], dim=-1)
        dh = torch.cat(
            [torch.ones_like(s), self.freq * torch.cos(a), -self.freq * torch.sin(a)],
            dim=-1,
        )
        return h, dh

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h, _ = self._features(s)
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)

    def forward_with_derivative(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N(s) and dN/ds in one pass (forward-mode differentiation by hand).

        For a scalar input the tangent of every layer is a vector of the
        layer's width: dh' = (1 - h'^2) * (W dh). One extra matmul per layer,
        against reverse mode's one backward pass per output component.
        """
        h, dh = self._features(s)
        for layer in self.hidden:
            h = torch.tanh(layer(h))
            dh = (1.0 - h * h) * (dh @ layer.weight.T)
        return self.out(h), dh @ self.out.weight.T


def derivative_fn(
    net: LorenzNet, mode: str
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """(N(s), dN/ds) by the named method."""
    if mode == "forward":
        return net.forward_with_derivative
    if mode == "jvp":
        from torch.func import jvp

        def f_jvp(s):
            return jvp(net, (s,), (torch.ones_like(s),))

        return f_jvp
    if mode == "autograd":

        def f_autograd(s):
            s = s.detach().requires_grad_(True)
            n = net(s)
            cols = [
                torch.autograd.grad(n[:, k].sum(), s, create_graph=True)[0]
                for k in range(n.shape[1])
            ]
            return n, torch.cat(cols, dim=-1)

        return f_autograd
    raise ValueError(f"unknown derivative mode {mode!r}")


@dataclass(frozen=True)
class PinnConfig:
    physics: bool = True
    width: int = 64
    depth: int = 4
    fourier: int = 8
    steps: int = 6000
    lr: float = 2e-3
    schedule: str = "cosine"  # "cosine" | "constant"
    warmup: int = 1000
    ramp: int = 1000
    w_phys: float = 1.0
    n_col: int = 256
    resample: bool = True
    lbfgs: int = 1000
    balance: str = "none"  # "none" | "gradnorm"
    balance_every: int = 100
    causal_eps: float = 0.0
    causal_chunks: int = 16
    derivative: str = "forward"
    compile: bool = False
    weight_decay: float = 0.0
    record_every: int = 25
    n_lbfgs_grid: int = 1024

    def label(self) -> str:
        return "pinn" if self.physics else "nn"


# The black box as tuned on the tuning seeds (see study.tune).
NN_DEFAULT = PinnConfig(
    physics=False, width=32, depth=3, fourier=0, lbfgs=0, warmup=0, ramp=0
)
PINN_DEFAULT = PinnConfig()


@dataclass
class LorenzFit:
    config: PinnConfig
    net: LorenzNet = field(repr=False)
    mu: np.ndarray
    sd: np.ndarray
    t_end: float
    theta: np.ndarray | None
    seconds: float
    history: dict[str, list] = field(repr=False)
    ms_per_step: float = float("nan")

    def _s(self, t: np.ndarray) -> torch.Tensor:
        s = 2.0 * np.asarray(t, float) / self.t_end - 1.0
        return torch.as_tensor(s, dtype=DTYPE)[:, None]

    def predict(self, t: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            n = self.net(self._s(t)).numpy()
        return self.mu + self.sd * n

    def derivative(self, t: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, dn = self.net.forward_with_derivative(self._s(t))
        return self.sd * dn.numpy() * (2.0 / self.t_end)

    def theta_dict(self) -> dict[str, float]:
        if self.theta is None:
            return {}
        return dict(zip(THETA_NAMES, map(float, self.theta), strict=True))


def _collocation(
    cfg: PinnConfig, gen: torch.Generator, fixed: torch.Tensor
) -> torch.Tensor:
    if not cfg.resample:
        return fixed
    # stratified: one uniform draw per equal-width bin, so every part of the
    # window is covered every step, in time order (the causal weights need it)
    n = cfg.n_col
    edges = torch.linspace(-1.0, 1.0, n + 1, dtype=DTYPE)
    u = torch.rand(n, generator=gen, dtype=DTYPE)
    return (edges[:-1] + u * (edges[1:] - edges[:-1]))[:, None]


def fit(obs: Observations, cfg: PinnConfig, seed: int) -> LorenzFit:
    """Train one network on one set of observations."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mu = obs.y_obs.mean(axis=0)
    sd = obs.y_obs.std(axis=0)
    require(bool(np.all(sd > 0)), "observations have a constant component")
    mu_t = torch.as_tensor(mu, dtype=DTYPE)
    sd_t = torch.as_tensor(sd, dtype=DTYPE)
    T = obs.t_end
    tscale = 2.0 / T

    net = LorenzNet(cfg.width, cfg.depth, cfg.fourier)
    raw = nn.Parameter(torch.zeros(3, dtype=DTYPE))
    theta0 = torch.as_tensor(THETA_INIT, dtype=DTYPE)
    s_obs = torch.as_tensor(2.0 * obs.t_obs / T - 1.0, dtype=DTYPE)[:, None]
    y_obs = torch.as_tensor(obs.y_obs, dtype=DTYPE)
    fixed = torch.linspace(-1.0, 1.0, cfg.n_col, dtype=DTYPE)[:, None]
    grid = torch.linspace(-1.0, 1.0, cfg.n_lbfgs_grid, dtype=DTYPE)[:, None]
    deriv = derivative_fn(net, cfg.derivative)
    if cfg.compile:
        deriv = torch.compile(deriv, dynamic=False)

    def theta() -> torch.Tensor:
        return theta0 * torch.exp(raw)

    def data_loss() -> torch.Tensor:
        return (((net(s_obs) * sd_t + mu_t - y_obs) / sd_t) ** 2).mean()

    def residual(s: torch.Tensor) -> torch.Tensor:
        """(n, 3) residual in network units."""
        n, dn = deriv(s)
        u = mu_t + sd_t * n
        th = theta()
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        f = torch.stack([th[0] * (y - x), x * (th[1] - z) - y, x * y - th[2] * z], -1)
        return (sd_t * dn * tscale - f) / (sd_t * tscale)

    def phys_loss(r: torch.Tensor) -> torch.Tensor:
        sq = (r**2).mean(dim=1)
        if cfg.causal_eps <= 0:
            return sq.mean()
        chunks = sq.reshape(cfg.causal_chunks, -1).mean(dim=1)
        with torch.no_grad():
            prior = torch.cumsum(chunks, 0) - chunks
            w = torch.exp(-cfg.causal_eps * prior)
        return (w * chunks).mean()

    weights = list(net.parameters())
    groups: list[dict[str, Any]] = [
        {"params": weights, "weight_decay": cfg.weight_decay}
    ]
    if cfg.physics:
        groups.append({"params": [raw], "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=cfg.lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr / 100)
        if cfg.schedule == "cosine"
        else None
    )
    hist: dict[str, list] = {
        k: []
        for k in (
            "step",
            "phase",
            "data",
            "phys",
            "phys_x",
            "phys_y",
            "phys_z",
            "w",
            "lr",
            *THETA_NAMES,
        )
    }

    def record(step, phase, ld, r, w, lr_now):
        hist["step"].append(step)
        hist["phase"].append(phase)
        hist["data"].append(float(ld))
        if r is None:
            for k in ("phys", "phys_x", "phys_y", "phys_z"):
                hist[k].append(float("nan"))
        else:
            per = (r.detach() ** 2).mean(dim=0)
            hist["phys"].append(float(per.mean()))
            for k, v in zip(("phys_x", "phys_y", "phys_z"), per, strict=True):
                hist[k].append(float(v))
        hist["w"].append(float(w))
        hist["lr"].append(float(lr_now))
        th = theta().detach().numpy() if cfg.physics else [float("nan")] * 3
        for k, v in zip(THETA_NAMES, th, strict=True):
            hist[k].append(float(v))

    def ramp_weight(step: int) -> float:
        if not cfg.physics:
            return 0.0
        if cfg.ramp <= 0:
            return 1.0 if step >= cfg.warmup else 0.0
        return min(1.0, max(0.0, (step - cfg.warmup) / cfg.ramp))

    balance = 1.0
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        w = cfg.w_phys * ramp_weight(step) * balance
        opt.zero_grad(set_to_none=True)
        ld = data_loss()
        r = residual(_collocation(cfg, gen, fixed)) if cfg.physics else None
        lp = phys_loss(r) if r is not None else None
        if (
            cfg.balance == "gradnorm"
            and lp is not None
            and w > 0
            and step % cfg.balance_every == 0
        ):
            gd = torch.autograd.grad(ld, weights, retain_graph=True)
            gp = torch.autograd.grad(lp, weights, retain_graph=True)
            nd = torch.sqrt(torch.stack([(g**2).sum() for g in gd]).sum())
            npg = torch.sqrt(torch.stack([(g**2).sum() for g in gp]).sum())
            balance = 0.9 * balance + 0.1 * float(nd / (npg + 1e-300)) / cfg.w_phys
            w = cfg.w_phys * ramp_weight(step) * balance
        loss = ld + w * lp if lp is not None else ld
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % cfg.record_every == 0 or step == cfg.steps - 1:
            record(step, "adam", ld.detach(), r, w, opt.param_groups[0]["lr"])
    adam_seconds = time.perf_counter() - t0
    ms_per_step = 1e3 * adam_seconds / max(cfg.steps, 1)

    if cfg.lbfgs > 0:
        w_final = cfg.w_phys * balance if cfg.physics else 0.0
        params = weights + ([raw] if cfg.physics else [])
        lb = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg.lbfgs,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )
        calls = [0]

        def closure():
            lb.zero_grad(set_to_none=True)
            ld = data_loss()
            if cfg.physics:
                r = residual(grid)
                loss = ld + w_final * (r**2).mean()
            else:
                r, loss = None, ld
            loss.backward()
            if calls[0] % cfg.record_every == 0:
                record(cfg.steps + calls[0], "lbfgs", ld.detach(), r, w_final, 1.0)
            calls[0] += 1
            return loss

        lb.step(closure)

    seconds = time.perf_counter() - t0
    th = theta().detach().numpy().copy() if cfg.physics else None
    return LorenzFit(cfg, net.eval(), mu, sd, T, th, seconds, hist, ms_per_step)


def config_dict(cfg: PinnConfig) -> dict:
    return asdict(cfg)


def with_changes(cfg: PinnConfig, **kw) -> PinnConfig:
    return replace(cfg, **kw)
