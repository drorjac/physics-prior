"""The gated hybrid: a law branch and a network branch fused by a learned gate.

    y = g y_M + (1 - g) y_D,   g = sigmoid(w . s + b)

y_M is the track's law with trainable constants, y_D an MLP of the inputs,
and s the standardised inputs together with y_M, y_D and y_M - y_D, all in
standardised units. It is the architecture of Jacoby et al. (ICASSP 2026),
built for rain from microwave links (`physprior.cml`), applied here to any
track whose law has a torch form.

Training is phased, which was the most stable scheme on the link data:

1. the law branch is fitted alone (`fit_physics`, least squares), the
   network branch alone (the `nn` arm's MLP and its tuned configuration);
2. the gate is trained with both branches frozen;
3. gate and network together;
4. everything, constants included, at a tenth of the learning rate.

With `ood=True` (the `hybrid_ood` arm, an extension that is not in the paper)
the network's share is also scaled by a trust factor exp(-(d / l)^2), d the
distance in standardised inputs to the nearest training input (for a
training input, to its nearest other one) and l three times the median of
those distances. Far from the data the output returns to the law:

    y = g' y_M + (1 - g') y_D,   g' = 1 - (1 - g) exp(-(d / l)^2)

Each stage minimises MSE of the fused output plus 0.1 times the MSE of each
branch (so neither branch starves when the gate saturates), in units of the
standard deviation of y. The constants it returns are the law branch's, so
the hybrid reports physics as the `physics` and `pinn` arms do.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .base import Fit, set_seed
from .neural import DTYPE, Standardiser, mlp
from .pinn import ParamSet, PhysParam, fit_physics

LAM = 0.1


class _Gated(torch.nn.Module):
    def __init__(self, law_t, params, net, std: Standardiser, d_in: int, anchors=None):
        super().__init__()
        # training inputs (standardised) and the trust length, for `ood`
        self.anchors = anchors
        self.ell = None
        if anchors is not None:
            d = torch.cdist(anchors, anchors)
            d.fill_diagonal_(float("inf"))
            self.ell = 3.0 * float(d.min(dim=1).values.median())
        self.law_t = law_t
        self.ps = ParamSet(params)
        self.net = net
        self.gate = torch.nn.Linear(d_in + 3, 1, dtype=DTYPE)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)
        self.mu_y, self.sd_y = std.mu_y, std.sd_y

    def forward(self, x_raw: torch.Tensor, x_std: torch.Tensor):
        ym = (self.law_t(x_raw, **self.ps.values()).reshape(-1) - self.mu_y) / self.sd_y
        yd = self.net(x_std).reshape(-1)
        s = torch.cat([x_std, ym[:, None], yd[:, None], (ym - yd)[:, None]], dim=1)
        g = torch.sigmoid(self.gate(s)).reshape(-1)
        if self.anchors is not None and self.ell:
            d = torch.cdist(x_std, self.anchors)
            # a training input is not its own neighbour
            d = torch.where(d < 1e-12, torch.full_like(d, float("inf")), d)
            trust = torch.exp(-((d.min(dim=1).values / self.ell) ** 2))
            g = 1 - (1 - g) * trust
        return g * ym + (1 - g) * yd, ym, yd, g


def _stage(model, params, xr, xs, ys, lr: float, epochs: int) -> None:
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y, ym, yd, _ = model(xr, xs)
        loss = (
            torch.mean((y - ys) ** 2)
            + LAM * torch.mean((ym - ys) ** 2)
            + LAM * torch.mean((yd - ys) ** 2)
        )
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()


def fit_hybrid(
    x: np.ndarray,
    y: np.ndarray,
    law_np,
    law_t,
    params: list[PhysParam],
    *,
    sigma: np.ndarray | None = None,
    seed: int = 0,
    width: int = 32,
    depth: int = 3,
    weight_decay: float = 1e-4,
    epochs: int = 4000,
    lr: float = 5e-3,
    ood: bool = False,
    name: str | None = None,
) -> Fit:
    t0 = time.time()
    set_seed(seed)
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[0] != len(y):
        x = x.T
    std = Standardiser.fit(x, y)
    xr = torch.tensor(x, dtype=DTYPE)
    xs = torch.tensor(std.x(x), dtype=DTYPE)
    ys = torch.tensor(std.y(y), dtype=DTYPE)

    # 1. the branches alone
    phys = fit_physics(x, y, law_np, params, sigma=sigma)
    start = [
        PhysParam(p.name, float(phys.params[p.name]), p.positive, p.lo, p.hi)
        for p in params
    ]
    net = mlp(x.shape[1], width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(xs).reshape(-1) - ys) ** 2)
        loss.backward()
        opt.step()
        sched.step()

    model = _Gated(law_t, start, net, std, x.shape[1], xs if ood else None)
    gate = list(model.gate.parameters())
    nets = list(model.net.parameters())
    consts = list(model.ps.parameters())
    # 2. gate only, 3. gate + network, 4. everything at lr / 10
    _stage(model, gate, xr, xs, ys, 1e-2, epochs // 4)
    _stage(model, gate + nets, xr, xs, ys, lr / 5, epochs // 4)
    _stage(model, gate + nets + consts, xr, xs, ys, lr / 10, epochs // 4)

    model.eval()

    def predict(xq: np.ndarray) -> np.ndarray:
        xq = np.atleast_2d(np.asarray(xq, float))
        if xq.shape[1] != x.shape[1]:
            xq = xq.T
        with torch.no_grad():
            out, _, _, _ = model(
                torch.tensor(xq, dtype=DTYPE), torch.tensor(std.x(xq), dtype=DTYPE)
            )
        return std.y_inv(out.numpy())

    def gate_at(xq: np.ndarray) -> np.ndarray:
        xq = np.atleast_2d(np.asarray(xq, float))
        if xq.shape[1] != x.shape[1]:
            xq = xq.T
        with torch.no_grad():
            _, _, _, g = model(
                torch.tensor(xq, dtype=DTYPE), torch.tensor(std.x(xq), dtype=DTYPE)
            )
        return g.numpy()

    with torch.no_grad():
        _, _, _, g_train = model(xr, xs)
    law = getattr(law_np, "expression", None)
    return Fit(
        name=name or ("hybrid_ood" if ood else "hybrid"),
        predict=predict,
        params=model.ps.numpy(),
        n_free=sum(p.numel() for p in model.parameters()),
        seconds=time.time() - t0,
        expression=f"g {law} + (1 - g) NN(x)" if law else "g law + (1 - g) NN(x)",
        extra={"gate_train_mean": float(g_train.mean()), "gate_at": gate_at},
    )
