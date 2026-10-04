"""The power-law branch, the GRU branch and the gated hybrid.

Power law (`PowerLaw`): for a link with k, alpha (ITU-R P.838-3 at its
frequency and polarisation) and path length L, each minute of the target bin
gives a rate (max(x - tau, 0) / (k L))^(1 / alpha) and the branch returns
their mean (Jacoby et al. 2026, eq. 4). tau is a per-link dead zone that
keeps measurement noise from being read as rain, initialised at three times
the link's robust noise sd and never below one quantisation step. With `trainable=True`, log k, alpha and log tau
are fitted per link by gradient descent.

GRU (`GRUBranch`): two GRU layers of 64 units over the history window, input
per step [x / L, x / 10], a linear read-out and a softplus, so the output is
a non-negative rate (the paper uses a ReLU; a softplus cannot die at
initialisation).

Hybrid (`Hybrid`): r = g r_M + (1 - g) r_D with g = sigmoid(w . s + b) and
s = [sd(x), mean(x), log(1 + r_M), log(1 + r_D), r_M - r_D], each feature
standardised with training-set statistics. The gate's weights and bias are
a shared part plus a per-link part (the paper fits one model per link).

Loss: MSE of the fused output, plus lambda_M and lambda_D times the MSE of
each branch so neither branch starves when the gate saturates.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

DTYPE = torch.float32


class PowerLaw(nn.Module):
    L: torch.Tensor

    def __init__(
        self,
        k: np.ndarray,
        alpha: np.ndarray,
        length_km: np.ndarray,
        tau_db: np.ndarray,
        m: int,
        trainable: bool = False,
    ):
        super().__init__()
        self.m = m
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DTYPE)  # noqa: E731
        self.register_buffer("L", t(length_km))
        self.log_k = nn.Parameter(torch.log(t(k)), requires_grad=trainable)
        self.alpha = nn.Parameter(t(alpha), requires_grad=trainable)
        self.log_tau = nn.Parameter(torch.log(t(tau_db)), requires_grad=trainable)

    def forward(self, x: torch.Tensor, link: torch.Tensor) -> torch.Tensor:
        xb = x[:, -self.m :]
        k = torch.exp(self.log_k[link])[:, None]
        a = self.alpha[link].clamp(0.4, 2.0)[:, None]
        tau = torch.exp(self.log_tau[link])[:, None]
        L = self.L[link][:, None]
        excess = torch.relu(xb - tau)
        eps = 1e-6
        r = (excess / (k * L) + eps) ** (1.0 / a) - eps ** (1.0 / a)
        return r.mean(dim=1)


class GRUBranch(nn.Module):
    L: torch.Tensor

    def __init__(self, length_km: np.ndarray, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.register_buffer("L", torch.as_tensor(np.asarray(length_km), dtype=DTYPE))
        self.gru = nn.GRU(2, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, link: torch.Tensor) -> torch.Tensor:
        L = self.L[link][:, None]
        z = torch.stack([x / L, x / 10.0], dim=-1)
        h, _ = self.gru(z)
        return nn.functional.softplus(self.out(h[:, -1])).squeeze(1)


def gate_features(x: torch.Tensor, rm: torch.Tensor, rd: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [x.std(dim=1), x.mean(dim=1), torch.log1p(rm), torch.log1p(rd), rm - rd], dim=1
    )


class Hybrid(nn.Module):
    s_mu: torch.Tensor
    s_sd: torch.Tensor

    def __init__(self, pl: PowerLaw, gru: GRUBranch, n_links: int):
        super().__init__()
        self.pl = pl
        self.gru = gru
        self.gate = nn.Linear(5, 1)
        # the paper fits one model per link, so each link has its own gate:
        # here a shared gate plus a per-link offset to its weights and bias
        self.link_w = nn.Embedding(n_links, 5)
        self.link_bias = nn.Embedding(n_links, 1)
        nn.init.zeros_(self.link_w.weight)
        nn.init.zeros_(self.link_bias.weight)
        self.register_buffer("s_mu", torch.zeros(5))
        self.register_buffer("s_sd", torch.ones(5))

    def set_feature_scale(self, x, link) -> None:
        with torch.no_grad():
            s = gate_features(x, self.pl(x, link), self.gru(x, link))
            self.s_mu.copy_(s.mean(0))
            self.s_sd.copy_(s.std(0).clamp_min(1e-6))

    def forward(self, x: torch.Tensor, link: torch.Tensor):
        rm = self.pl(x, link)
        rd = self.gru(x, link)
        s = (gate_features(x, rm, rd) - self.s_mu) / self.s_sd
        z = self.gate(s) + (self.link_w(link) * s).sum(1, keepdim=True)
        g = torch.sigmoid(z + self.link_bias(link)).squeeze(1)
        return g * rm + (1 - g) * rd, rm, rd, g


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def robust_sd(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))
