"""The physics-informed arms.

Three of them, because "physics-informed" is not one thing:

`oracle`    the textbook law with the PUBLISHED constants. No fitting at all.
            The ceiling: no method that learns from this data should beat it.

`physics`   the textbook law with its constants FITTED. This is the classical
            inverse problem, and it is the arm that returns a number a
            physicist can compare with the literature, with an error bar.

`pinn`      the law plus a neural correction,

                y(x) = law(x; theta) + sd_y * NN(x)

            trained on

                L = MSE(y_pred, y)/sd_y^2  +  w_phys * mean(NN(x)^2)

            with `theta` trainable. `w_phys` is the dial this project turns:
            at w_phys -> infinity the correction is crushed and the arm is
            `physics`; at w_phys = 0 the network is free and the arm is a
            black box wearing a physics hat. Sweeping it measures what the
            physics term is worth, in the loss, in units the data sets.

A fourth, `pinn_ode`, is used where the law is a differential equation: the
network represents the solution and the residual of the ODE -- with the
physical parameter trainable -- is added to the loss. That is the textbook
PINN of Raissi et al. (2019).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import curve_fit

from .base import Fit, set_seed
from .neural import DTYPE, Standardiser, mlp


@dataclass
class PhysParam:
    """A physical constant to be recovered.

    Positive constants are optimised in log space: it conditions the problem,
    enforces positivity, and makes the prior scale-free over decades.
    """

    name: str
    init: float
    positive: bool = True
    lo: float = -np.inf
    hi: float = np.inf


class ParamSet(torch.nn.Module):
    def __init__(self, params: list[PhysParam]):
        super().__init__()
        self.specs = params
        self.raw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((), dtype=DTYPE)) for _ in params]
        )

    def values(self) -> dict[str, torch.Tensor]:
        out = {}
        for spec, r in zip(self.specs, self.raw, strict=True):
            out[spec.name] = (
                spec.init * torch.exp(r) if spec.positive else spec.init + r
            )
        return out

    def numpy(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in self.values().items()}


# ---------------------------------------------------------------------------
# oracle and physics
# ---------------------------------------------------------------------------


def oracle(law_np, theta: dict[str, float]) -> Fit:
    """The law with published constants. Nothing is fitted."""
    return Fit(
        name="oracle",
        predict=lambda xq: law_np(xq, **theta),
        params=dict(theta),
        n_free=0,
        expression=getattr(law_np, "expression", None),
    )


def fit_physics(
    x: np.ndarray,
    y: np.ndarray,
    law_np,
    params: list[PhysParam],
    sigma: np.ndarray | None = None,
    name: str = "physics",
) -> Fit:
    """Least squares on the closed-form law. Returns constants with error bars."""
    t0 = time.time()
    names = [p.name for p in params]
    p0 = [p.init for p in params]
    bounds = ([p.lo for p in params], [p.hi for p in params])

    def f(xq, *theta):
        return np.asarray(
            law_np(xq, **dict(zip(names, theta, strict=True))), float
        ).ravel()

    popt, pcov = curve_fit(
        f,
        x,
        np.asarray(y, float).ravel(),
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=200000,
    )
    perr = np.sqrt(np.diag(pcov))
    theta = dict(zip(names, popt, strict=True))
    return Fit(
        name=name,
        predict=lambda xq: f(xq, *popt),
        params=theta,
        param_sigma=dict(zip(names, perr, strict=True)),
        n_free=len(params),
        seconds=time.time() - t0,
        expression=getattr(law_np, "expression", None),
    )


# ---------------------------------------------------------------------------
# pinn: law + neural correction, with the physics weight as a dial
# ---------------------------------------------------------------------------


def fit_pinn(
    x: np.ndarray,
    y: np.ndarray,
    law_t,
    params: list[PhysParam],
    *,
    w_phys: float = 1.0,
    width: int = 32,
    depth: int = 3,
    epochs: int = 6000,
    lr: float = 5e-3,
    seed: int = 0,
    weight_decay: float = 0.0,
    name: str = "pinn",
    record_every: int = 0,
    record_grid: np.ndarray | None = None,
) -> Fit:
    """`record_every > 0` keeps the history: the two loss terms separately,
    every trainable constant, and the prediction on `record_grid`. Watching the
    constant walk toward its published value is the clearest picture in this
    project of what the physics term in the loss is doing."""
    t0 = time.time()
    set_seed(seed)
    std = Standardiser.fit(x, y)
    xs = torch.tensor(std.x(x), dtype=DTYPE)
    xt = torch.tensor(
        np.atleast_2d(np.asarray(x, float)).reshape(len(y), -1), dtype=DTYPE
    )
    yt = torch.tensor(np.asarray(y, float).ravel(), dtype=DTYPE)

    net = mlp(xs.shape[1], width, depth)
    ps = ParamSet(params)
    opt = torch.optim.Adam(
        [
            {"params": net.parameters(), "weight_decay": weight_decay},
            {"params": ps.parameters(), "weight_decay": 0.0},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: dict[str, list] = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "phys_loss": [],
        "pred": [],
    }
    for spec in params:
        history[spec.name] = []
    grid_np = None
    if record_every and record_grid is not None:
        g = np.asarray(record_grid, float)
        grid_np = g.reshape(-1, 1) if g.ndim == 1 else g

    for ep in range(epochs):
        opt.zero_grad()
        corr = net(xs).squeeze(-1)
        pred = law_t(xt, **ps.values()).squeeze() + std.sd_y * corr
        data = torch.mean(((pred - yt) / std.sd_y) ** 2)
        phys = torch.mean(corr**2)
        (data + w_phys * phys).backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["data_loss"].append(float(data.detach()))
            history["phys_loss"].append(float(phys.detach()))
            history["loss"].append(float((data + w_phys * phys).detach()))
            for k, v in ps.numpy().items():
                history[k].append(v)
            if grid_np is not None:
                with torch.no_grad():
                    c = net(torch.tensor(std.x(grid_np), dtype=DTYPE)).squeeze(-1)
                    pq = law_t(
                        torch.tensor(grid_np, dtype=DTYPE), **ps.values()
                    ).squeeze()
                    history["pred"].append((pq + std.sd_y * c).numpy().ravel())

    theta = ps.numpy()

    def predict(xq: np.ndarray) -> np.ndarray:
        xq2 = np.atleast_2d(np.asarray(xq, float))
        if xq2.shape[1] != xs.shape[1]:
            xq2 = xq2.T
        with torch.no_grad():
            c = net(torch.tensor(std.x(xq2), dtype=DTYPE)).squeeze(-1)
            p = law_t(torch.tensor(xq2, dtype=DTYPE), **ps.values()).squeeze()
            return (p + std.sd_y * c).numpy().ravel()

    return Fit(
        name=name,
        predict=predict,
        params=theta,
        n_free=len(params) + sum(p.numel() for p in net.parameters()),
        seconds=time.time() - t0,
        history=history if record_every else None,
        extra={
            "w_phys": w_phys,
            "correction_rms_frac": float(
                torch.sqrt(torch.mean(net(xs).squeeze(-1) ** 2)).detach()
            ),
            "data_mse_std_units": float(data.detach()),
        },
    )


# ---------------------------------------------------------------------------
# pinn_ode: the textbook PINN. Network is the solution; the ODE is the loss.
# ---------------------------------------------------------------------------


def fit_pinn_ode(
    t: np.ndarray,
    y: np.ndarray,
    residual_t,
    params: list[PhysParam],
    *,
    w_phys: float = 1.0,
    n_collocation: int = 400,
    width: int = 32,
    depth: int = 3,
    epochs: int = 8000,
    lr: float = 5e-3,
    seed: int = 0,
    t_domain: tuple[float, float] | None = None,
    name: str = "pinn_ode",
    record_every: int = 0,
    record_grid: np.ndarray | None = None,
) -> Fit:
    """`residual_t(t, y, dy_dt, **theta)` must vanish when the ODE holds."""
    t0 = time.time()
    set_seed(seed)
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    std = Standardiser.fit(t.reshape(-1, 1), y)
    lo, hi = t_domain if t_domain is not None else (t.min(), t.max())

    tt = torch.tensor(std.x(t.reshape(-1, 1)), dtype=DTYPE)
    yt = torch.tensor(std.y(y), dtype=DTYPE).reshape(-1, 1)
    tc_raw = np.linspace(lo, hi, n_collocation).reshape(-1, 1)
    tc = torch.tensor(std.x(tc_raw), dtype=DTYPE, requires_grad=True)

    net = mlp(1, width, depth)
    ps = ParamSet(params)
    opt = torch.optim.Adam(list(net.parameters()) + list(ps.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: dict[str, list] = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "phys_loss": [],
        "pred": [],
    }
    for spec in params:
        history[spec.name] = []
    grid_t = None
    if record_every and record_grid is not None:
        grid_t = torch.tensor(
            std.x(np.asarray(record_grid, float).reshape(-1, 1)), dtype=DTYPE
        )

    for ep in range(epochs):
        opt.zero_grad()
        data = torch.mean((net(tt) - yt) ** 2)
        yc_std = net(tc)
        (dyc_std,) = torch.autograd.grad(
            yc_std, tc, torch.ones_like(yc_std), create_graph=True
        )
        # de-standardise back to physical units before the ODE sees them
        yc = yc_std * std.sd_y + std.mu_y
        dyc = dyc_std * std.sd_y / std.sd_x[0]
        t_phys = tc * std.sd_x[0] + std.mu_x[0]
        res = residual_t(
            t_phys.squeeze(-1), yc.squeeze(-1), dyc.squeeze(-1), **ps.values()
        )
        phys = torch.mean(res**2)
        (data + w_phys * phys).backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["data_loss"].append(float(data.detach()))
            history["phys_loss"].append(float(phys.detach()))
            history["loss"].append(float((data + w_phys * phys).detach()))
            for k, v in ps.numpy().items():
                history[k].append(v)
            if grid_t is not None:
                with torch.no_grad():
                    history["pred"].append(std.y_inv(net(grid_t).numpy().ravel()))

    def predict(tq: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = (
                net(
                    torch.tensor(
                        std.x(np.asarray(tq, float).reshape(-1, 1)), dtype=DTYPE
                    )
                )
                .numpy()
                .ravel()
            )
        return std.y_inv(z)

    return Fit(
        name=name,
        predict=predict,
        params=ps.numpy(),
        n_free=len(params) + sum(p.numel() for p in net.parameters()),
        seconds=time.time() - t0,
        history=history if record_every else None,
        extra={
            "w_phys": w_phys,
            "data_mse_std_units": float(data.detach()),
            "phys_residual": float(phys.detach()),
        },
    )
