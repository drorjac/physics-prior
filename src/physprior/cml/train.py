"""Training every arm on one dataset, one seed, one data budget.

`fit_arms` trains the branches once and builds the hybrids from them, so the
gate-only and phased hybrids start from exactly the `pl_cal` and `gru` models
that are themselves reported:

    pl_itu          no training
    pl_cal          k, alpha, tau per link                 (Adam, lr 1e-2)
    gru             the GRU branch alone                    (Adam, lr 1e-3)
    hybrid_joint    PL (from ITU) + new GRU + gate, together
    hybrid_gate     pl_cal and gru frozen, gate trained
    hybrid_phased   gate, then gate + GRU, then everything at lr / 10

For the two hybrids built on pretrained branches, each link's gate starts
at that link's least-squares blend of the frozen branches on the training
bins (`init_link_gates`), so a link whose power law is useless starts with
the gate near 0 instead of at 0.5.

Each epoch draws up to 20 000 wet training bins and as many dry ones, at
random, and weights each by the size of its class over the number drawn, so
the training loss is an unbiased estimate of the loss over the natural mix
of wet and dry bins (about 95 % dry); validation and test use every bin. Each stage stops
early on the validation loss (patience 3, at most 15 epochs) and keeps its best state.
Hyperparameters are fixed in `TrainConfig` before any result was seen.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import WET_MMH, Samples
from .models import DTYPE, GRUBranch, Hybrid, PowerLaw, robust_sd

ARMS = (
    "pl_itu",
    "pl_cal",
    "gru",
    "hybrid_joint",
    "hybrid_gate",
    "hybrid_phased",
)


@dataclass(frozen=True)
class TrainConfig:
    batch: int = 256
    epochs: int = 15
    patience: int = 3
    lr: float = 1e-3
    lr_pl: float = 1e-2
    weight_decay: float = 1e-5
    lam_m: float = 0.1
    lam_d: float = 0.1
    dry_ratio: float = 1.0
    max_wet: int = 20_000
    tau_sigmas: float = 3.0
    tau_floor_db: float = 0.15
    max_val: int = 30_000

    def quick(self) -> TrainConfig:
        return TrainConfig(epochs=2, patience=1, max_val=5_000)


def _t(a, dtype=DTYPE) -> torch.Tensor:
    return torch.as_tensor(a, dtype=dtype)


def link_noise(s: Samples, train: np.ndarray) -> np.ndarray:
    """Robust sd of the excess attenuation per link on the training samples,
    without labels: most minutes are dry, so the median absolute deviation
    measures the noise."""
    out = np.full(len(s.links), np.nan)
    for i in range(len(s.links)):
        v = s.x[train & (s.link == i), -s.m :].ravel()
        out[i] = robust_sd(v)
    return out


class _Data:
    def __init__(self, s: Samples, idx: np.ndarray):
        self.x = _t(s.x[idx])
        self.r = _t(s.r[idx])
        self.link = _t(s.link[idx], torch.long)

    def __len__(self) -> int:
        return len(self.r)


def _batches(d: _Data, rng: np.random.Generator, cfg: TrainConfig):
    wet = np.flatnonzero(d.r.numpy() > WET_MMH)
    dry = np.flatnonzero(d.r.numpy() <= WET_MMH)
    if len(wet) > cfg.max_wet:
        wet = rng.choice(wet, cfg.max_wet, replace=False)
    n_dry = min(len(dry), int(cfg.dry_ratio * max(len(wet), 1)))
    n_wet_all = int((d.r.numpy() > WET_MMH).sum())
    idx = np.concatenate([wet, rng.choice(dry, n_dry, replace=False)])
    # importance weights: each drawn bin stands for all the bins of its class,
    # so the expected loss is the loss over the natural wet/dry mix
    w = np.concatenate(
        [
            np.full(len(wet), n_wet_all / max(len(wet), 1)),
            np.full(n_dry, len(dry) / max(n_dry, 1)),
        ]
    )
    w = w / w.mean()
    perm = rng.permutation(len(idx))
    idx, w = idx[perm], w[perm]
    for i in range(0, len(idx), cfg.batch):
        j = _t(idx[i : i + cfg.batch], torch.long)
        yield d.x[j], d.r[j], d.link[j], _t(w[i : i + cfg.batch])


def _predict(model, d: _Data, hybrid: bool, bs: int = 8192):
    outs: list[list[np.ndarray]] = [[], [], [], []]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(d), bs):
            x, lk = d.x[i : i + bs], d.link[i : i + bs]
            if hybrid:
                r, rm, rd, g = model(x, lk)
                for o, v in zip(outs, (r, rm, rd, g), strict=True):
                    o.append(v.numpy())
            else:
                outs[0].append(model(x, lk).numpy())
    return [np.concatenate(o) if o else None for o in outs]


def _loss(model, x, r, lk, hybrid: bool, cfg: TrainConfig, w=None) -> torch.Tensor:
    def mse(a, b):
        e = (a - b) ** 2
        return e.mean() if w is None else (w * e).sum() / w.sum()

    if not hybrid:
        return mse(model(x, lk), r)
    rh, rm, rd, _ = model(x, lk)
    return mse(rh, r) + cfg.lam_m * mse(rm, r) + cfg.lam_d * mse(rd, r)


def _fit(
    model: nn.Module,
    params: Sequence[torch.Tensor],
    tr: _Data,
    va: _Data,
    lr: float,
    hybrid: bool,
    cfg: TrainConfig,
    rng: np.random.Generator,
) -> list[dict]:
    opt = torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
    # the starting state is a candidate too: a stage that only makes the
    # validation loss worse returns the model it was given
    model.eval()
    with torch.no_grad():
        best = float(_loss(model, va.x, va.r, va.link, hybrid, cfg))
    best_state, wait, hist = copy.deepcopy(model.state_dict()), 0, []
    for ep in range(cfg.epochs):
        model.train()
        tl, nb = 0.0, 0
        for x, r, lk, w in _batches(tr, rng, cfg):
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, x, r, lk, hybrid, cfg, w)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tl += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vl = float(_loss(model, va.x, va.r, va.link, hybrid, cfg))
        hist.append({"epoch": ep, "train": tl / max(nb, 1), "val": vl})
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return hist


def metrics(pred: np.ndarray, r: np.ndarray, link: np.ndarray | None = None) -> dict:
    e = pred - r
    mu = float(r.mean())
    wet = r > WET_MMH
    out = {
        "nrmse": float(np.sqrt(np.mean(e**2)) / mu),
        "nbias": float(e.mean() / mu),
        "corr": float(np.corrcoef(pred, r)[0, 1]) if pred.std() > 0 else float("nan"),
        "mean_ref": mu,
    }
    if link is not None:
        # robust to one bad link: the median over links of the per-link NRMSE
        per = []
        for i in np.unique(link):
            m = link == i
            if r[m].mean() > 0:
                per.append(np.sqrt(np.mean(e[m] ** 2)) / r[m].mean())
        out["nrmse_link_median"] = float(np.median(per)) if per else float("nan")
    if wet.any():
        mw = float(r[wet].mean())
        out["nrmse_wet"] = float(np.sqrt(np.mean(e[wet] ** 2)) / mw)
        out["nbias_wet"] = float(e[wet].mean() / mw)
    return out


def fit_arms(
    s: Samples,
    seed: int,
    train_mask: np.ndarray | None = None,
    cfg: TrainConfig = TrainConfig(),
    arms: tuple[str, ...] = ARMS,
) -> dict:
    """Train the requested arms; return test metrics and test predictions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    train = s.split == 0 if train_mask is None else train_mask
    val_idx = np.flatnonzero(s.split == 1)
    if len(val_idx) > cfg.max_val:
        val_idx = np.sort(np.random.default_rng(0).choice(val_idx, cfg.max_val, False))
    tr = _Data(s, np.flatnonzero(train))
    va = _Data(s, val_idx)
    te = _Data(s, np.flatnonzero(s.split == 2))

    L = s.links["length_km"].to_numpy(float)
    k0 = s.links["k_itu"].to_numpy(float)
    a0 = s.links["alpha_itu"].to_numpy(float)
    sd = link_noise(s, train)
    # the dead zone is three noise sd, and never less than one quantisation
    # step: below that a single flicker of the last digit would read as rain
    quant = s.links["quant_db"].to_numpy(float) if "quant_db" in s.links else 0.0
    tau0 = np.maximum(
        cfg.tau_sigmas
        * np.maximum(np.nan_to_num(sd, nan=cfg.tau_floor_db), cfg.tau_floor_db),
        quant,
    )

    def pl(trainable: bool) -> PowerLaw:
        return PowerLaw(k0, a0, L, tau0, s.m, trainable)

    out: dict = {"metrics": {}, "pred": {}, "gate": {}, "seconds": {}, "params": {}}

    def record(name: str, model: nn.Module, hybrid: bool, t0: float) -> None:
        p = _predict(model, te, hybrid)
        out["pred"][name] = p[0]
        out["metrics"][name] = metrics(p[0], te.r.numpy(), te.link.numpy())
        if hybrid:
            out["gate"][name] = p[3]
            out["metrics"][name]["gate_mean"] = float(p[3].mean())
            wet = te.r.numpy() > WET_MMH
            out["metrics"][name]["gate_wet"] = (
                float(p[3][wet].mean()) if wet.any() else np.nan
            )
            out["metrics"][name]["nrmse_pl_branch"] = metrics(p[1], te.r.numpy())[
                "nrmse"
            ]
            out["metrics"][name]["nrmse_gru_branch"] = metrics(p[2], te.r.numpy())[
                "nrmse"
            ]
        out["seconds"][name] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if "pl_itu" in arms:
        record("pl_itu", pl(False), False, t0)

    need_cal = {"pl_cal", "hybrid_gate", "hybrid_phased"} & set(arms)
    need_gru = {"gru", "hybrid_gate", "hybrid_phased"} & set(arms)
    m_cal = m_gru = None
    if need_cal:
        t0 = time.perf_counter()
        m_cal = pl(True)
        _fit(m_cal, list(m_cal.parameters()), tr, va, cfg.lr_pl, False, cfg, rng)
        if "pl_cal" in arms:
            record("pl_cal", m_cal, False, t0)
        out["params"]["pl_cal"] = {
            "k": torch.exp(m_cal.log_k).detach().numpy().tolist(),
            "alpha": m_cal.alpha.detach().numpy().tolist(),
            "tau": torch.exp(m_cal.log_tau).detach().numpy().tolist(),
        }
    if need_gru:
        t0 = time.perf_counter()
        m_gru = GRUBranch(L)
        _fit(m_gru, list(m_gru.parameters()), tr, va, cfg.lr, False, cfg, rng)
        if "gru" in arms:
            record("gru", m_gru, False, t0)

    if "hybrid_joint" in arms:
        t0 = time.perf_counter()
        h = Hybrid(pl(True), GRUBranch(L), len(s.links))
        h.set_feature_scale(tr.x[:20000], tr.link[:20000])
        groups = [
            {"params": list(h.pl.parameters()), "lr": cfg.lr_pl},
            {
                "params": list(h.gru.parameters())
                + list(h.gate.parameters())
                + list(h.link_bias.parameters())
                + list(h.link_w.parameters())
            },
        ]
        _fit_groups(h, groups, tr, va, cfg, rng)
        record("hybrid_joint", h, True, t0)

    if {"hybrid_gate", "hybrid_phased"} & set(arms):
        assert m_cal is not None and m_gru is not None
        for name in ("hybrid_gate", "hybrid_phased"):
            if name not in arms:
                continue
            t0 = time.perf_counter()
            h = Hybrid(copy.deepcopy(m_cal), copy.deepcopy(m_gru), len(s.links))
            h.set_feature_scale(tr.x[:20000], tr.link[:20000])
            for p in h.pl.parameters():
                p.requires_grad_(False)
            for p in h.gru.parameters():
                p.requires_grad_(False)
            init_link_gates(h, tr, len(s.links))
            _fit(
                h,
                list(h.gate.parameters())
                + list(h.link_bias.parameters())
                + list(h.link_w.parameters()),
                tr,
                va,
                1e-2,
                True,
                cfg,
                rng,
            )
            if name == "hybrid_phased":
                for p in h.gru.parameters():
                    p.requires_grad_(True)
                _fit(
                    h,
                    list(h.gru.parameters())
                    + list(h.gate.parameters())
                    + list(h.link_bias.parameters())
                    + list(h.link_w.parameters()),
                    tr,
                    va,
                    cfg.lr,
                    True,
                    cfg,
                    rng,
                )
                for p in h.pl.parameters():
                    p.requires_grad_(True)
                groups = [
                    {"params": list(h.pl.parameters()), "lr": cfg.lr_pl / 10},
                    {
                        "params": list(h.gru.parameters())
                        + list(h.gate.parameters())
                        + list(h.link_bias.parameters())
                        + list(h.link_w.parameters()),
                        "lr": cfg.lr / 10,
                    },
                ]
                _fit_groups(h, groups, tr, va, cfg, rng)
            record(name, h, True, t0)
    out["n_train"] = len(tr)
    out["n_test"] = len(te)
    out["noise_sd"] = sd.tolist()
    return out


def init_link_gates(h: Hybrid, tr: _Data, n_links: int) -> None:
    """Start each link's gate at its best fixed blend of the two frozen
    branches on the training bins, g* = argmin sum (g r_M + (1 - g) r_D - r)^2
    = sum (r_M - r_D)(r - r_D) / sum (r_M - r_D)^2, clipped to [0.001, 0.999].
    The gate's shared weights start at zero, so the gate starts at g*."""
    rm, rd = [], []
    h.eval()
    with torch.no_grad():
        for i in range(0, len(tr), 8192):
            x, lk = tr.x[i : i + 8192], tr.link[i : i + 8192]
            rm.append(h.pl(x, lk))
            rd.append(h.gru(x, lk))
        m = torch.cat(rm).numpy()
        d = torch.cat(rd).numpy()
    r = tr.r.numpy()
    links = tr.link.numpy()
    num = np.bincount(links, weights=(m - d) * (r - d), minlength=n_links)
    den = np.bincount(links, weights=(m - d) ** 2, minlength=n_links)
    g = np.where(den > 0, num / np.maximum(den, 1e-12), 0.5)
    g = np.clip(g, 1e-3, 1 - 1e-3)
    with torch.no_grad():
        h.gate.weight.zero_()
        h.gate.bias.zero_()
        h.link_bias.weight.copy_(_t(np.log(g / (1 - g)))[:, None])


def _fit_groups(model, groups, tr, va, cfg: TrainConfig, rng) -> list[dict]:
    """`_fit` with per-group learning rates."""
    opt = torch.optim.Adam(
        [{**g, "lr": g.get("lr", cfg.lr)} for g in groups],
        weight_decay=cfg.weight_decay,
    )
    params = [p for g in groups for p in g["params"]]
    # the starting state is a candidate too: a stage that only makes the
    # validation loss worse returns the model it was given
    model.eval()
    with torch.no_grad():
        best = float(_loss(model, va.x, va.r, va.link, True, cfg))
    best_state, wait, hist = copy.deepcopy(model.state_dict()), 0, []
    for ep in range(cfg.epochs):
        model.train()
        for x, r, lk, w in _batches(tr, rng, cfg):
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, x, r, lk, True, cfg, w)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(_loss(model, va.x, va.r, va.link, True, cfg))
        hist.append({"epoch": ep, "val": vl})
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return hist
