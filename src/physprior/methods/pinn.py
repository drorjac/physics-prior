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


@dataclass(frozen=True)
class PinnOptions:
    """Switches for the `pinn` arm. Every one of them defaults to OFF.

    The arm that produced the committed results is `PinnOptions()`, so an
    option can only change a published number by being turned on deliberately
    and shown, on the TUNING seeds, to earn it (invariant 6). `tag` names the
    configuration in an ablation table.

    early_stopping  carve a validation split out of the TRAINING data and
                    keep the epoch that minimised its data loss. The
                    correction network is what overfits -- where the law is
                    already right the ideal correction is zero, which a
                    network cannot represent, so it fits noise instead and
                    the damage only shows up off the training range.
    lbfgs           a second-order refinement after Adam. Adam gets close on
                    a badly scaled problem; L-BFGS is what finishes.
    ensemble        train N members from consecutive seeds and average them.
                    Gives the arm a spread over seeds, which is the only
                    uncertainty it has to set against `curve_fit`'s sigma.
    fourier         random Fourier features on the correction's input, for
                    spectral bias. A prior in its own right: it assumes the
                    residual has structure at the scale it encodes.
    balance         learning-rate annealing (Wang et al. 2021): rescale the
                    physics weight from the ratio of gradient norms, so the
                    two loss terms contribute comparably. Note this makes
                    w_phys adaptive, which is exactly what the w_phys sweep
                    holds fixed -- the two are alternatives, not a stack.
    """

    early_stopping: bool = False
    val_frac: float = 0.25
    patience: int = 400
    check_every: int = 25
    lbfgs: bool = False
    lbfgs_steps: int = 250
    ensemble: int = 1
    fourier: int = 0
    fourier_scale: float = 1.0
    balance: bool = False
    balance_every: int = 100
    balance_alpha: float = 0.9

    @property
    def tag(self) -> str:
        on = []
        if self.early_stopping:
            on.append("early")
        if self.lbfgs:
            on.append("lbfgs")
        if self.ensemble > 1:
            on.append(f"ens{self.ensemble}")
        if self.fourier:
            on.append(f"fourier{self.fourier}")
        if self.balance:
            on.append("balance")
        return "+".join(on) if on else "baseline"


DEFAULT_PINN = PinnOptions()

# The configuration the REPORTED results are produced with.
#
# History. On 2026-09-23 `balance` was frozen in after an ablation on the
# tuning seeds showed it helping on five of eight cells. On 2026-09-27 it was
# taken out again, for a reason the ablation could not see: in this arm the
# physics term is the penalty mean(NN^2), whose gradient vanishes as the
# correction shrinks. Gradient-norm annealing sets the weight to
# |grad data| / |grad penalty|, so the smaller the correction the larger the
# weight, without bound -- measured from several hundred to 1e8 across the
# tracks and still rising at the end of training, with the correction
# switched off and the arm reduced to the `physics` fit. The
# ablation's gain was that collapse, measured against an unbalanced arm whose
# correction overfits (docs/optimization/ section 5). Annealing is meant for
# PDE residuals that do not vanish with the network; it is still available
# through PinnOptions, and not on by default.
#
# In its place each track carries its own fixed weight, `Problem.pinn_w_phys`,
# chosen on the tuning seeds by `physprior.benchmark.pinn_tuning` on a
# validation block cut from the top of the track's own training range. That
# makes w_phys a tuned hyperparameter of the arm, as width and weight decay
# are for `nn`.
#
# Early stopping and Fourier features were measured and REJECTED: early
# stopping hurt four cells and helped none, Fourier features made hydrogen
# interpolation 98x worse. L-BFGS ran on every track and its proposal never
# lowered the loss, so the revert guard discarded it every time. A 5-member
# ensemble helps modestly for five times the compute and stays available for
# where an uncertainty on the recovered constant is the point.
FROZEN_PINN = DEFAULT_PINN

# Below this many training points a validation split cannot be carved without
# leaving the fit with too little to fit -- track G trains on as few as two.
MIN_POINTS_FOR_EARLY_STOPPING = 6


class FourierFeatures(torch.nn.Module):
    """x -> [sin(2 pi B x), cos(2 pi B x)], B fixed random.

    The frequencies are drawn once and frozen: they are part of the model's
    prior, not something the optimiser gets to choose.
    """

    def __init__(self, d_in: int, n: int, scale: float, generator):
        super().__init__()
        b = torch.randn(d_in, n, generator=generator, dtype=DTYPE) * scale
        self.register_buffer("freqs", b)

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _correction_net(d_in: int, width: int, depth: int, opts: PinnOptions, seed: int):
    if not opts.fourier:
        return mlp(d_in, width, depth)
    gen = torch.Generator().manual_seed(seed)
    ff = FourierFeatures(d_in, opts.fourier, opts.fourier_scale, gen)
    return torch.nn.Sequential(ff, mlp(2 * opts.fourier, width, depth))


def _split_validation(n: int, frac: float, seed: int):
    """A validation split carved from the TRAINING data only.

    Never from the held-out set: selecting an epoch on the data the arm is
    scored against is the same error as tuning on a reporting seed.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, round(frac * n))
    return perm[n_val:], perm[:n_val]


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
    options: PinnOptions = DEFAULT_PINN,
) -> Fit:
    """`record_every > 0` keeps the history: the two loss terms separately,
    every trainable constant, and the prediction on `record_grid`. Watching the
    constant walk toward its published value is the clearest picture in this
    project of what the physics term in the loss is doing.

    With `options.ensemble > 1` the members are trained from consecutive
    seeds and averaged; the spread of the recovered constants across members
    is reported as `param_sigma`, which is the arm's only error bar.
    """
    if options.ensemble <= 1:
        return _fit_pinn_once(
            x,
            y,
            law_t,
            params,
            w_phys=w_phys,
            width=width,
            depth=depth,
            epochs=epochs,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            name=name,
            record_every=record_every,
            record_grid=record_grid,
            options=options,
        )

    t0 = time.time()
    members = [
        _fit_pinn_once(
            x,
            y,
            law_t,
            params,
            w_phys=w_phys,
            width=width,
            depth=depth,
            epochs=epochs,
            lr=lr,
            seed=seed + k,
            weight_decay=weight_decay,
            name=name,
            record_every=record_every if k == 0 else 0,
            record_grid=record_grid,
            options=options,
        )
        for k in range(options.ensemble)
    ]

    def predict(xq: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(xq) for m in members], axis=0)

    names = members[0].params.keys()
    theta = {k: float(np.mean([m.params[k] for m in members])) for k in names}
    sigma = {k: float(np.std([m.params[k] for m in members], ddof=1)) for k in names}
    return Fit(
        name=name,
        predict=predict,
        params=theta,
        param_sigma=sigma,
        n_free=sum(m.n_free for m in members),
        seconds=time.time() - t0,
        history=members[0].history,
        extra={
            "w_phys": w_phys,
            "options": options.tag,
            "ensemble": options.ensemble,
            "engaged": all(m.extra.get("engaged", True) for m in members),
            "members": [m.params for m in members],
        },
    )


def _fit_pinn_once(
    x: np.ndarray,
    y: np.ndarray,
    law_t,
    params: list[PhysParam],
    *,
    w_phys: float,
    width: int,
    depth: int,
    epochs: int,
    lr: float,
    seed: int,
    weight_decay: float,
    name: str,
    record_every: int,
    record_grid: np.ndarray | None,
    options: PinnOptions,
) -> Fit:
    t0 = time.time()
    set_seed(seed)
    std = Standardiser.fit(x, y)
    xs = torch.tensor(std.x(x), dtype=DTYPE)
    xt = torch.tensor(
        np.atleast_2d(np.asarray(x, float)).reshape(len(y), -1), dtype=DTYPE
    )
    yt = torch.tensor(np.asarray(y, float).ravel(), dtype=DTYPE)

    use_early = options.early_stopping and len(y) >= MIN_POINTS_FOR_EARLY_STOPPING
    if use_early:
        i_fit, i_val = _split_validation(len(y), options.val_frac, seed)
        fit_idx = torch.tensor(i_fit, dtype=torch.long)
        val_idx = torch.tensor(i_val, dtype=torch.long)
    else:
        fit_idx = torch.arange(len(y))
        val_idx = None

    net = _correction_net(xs.shape[1], width, depth, options, seed)
    ps = ParamSet(params)
    opt = torch.optim.Adam(
        [
            {"params": net.parameters(), "weight_decay": weight_decay},
            {"params": ps.parameters(), "weight_decay": 0.0},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def losses(idx):
        corr = net(xs[idx]).squeeze(-1)
        pred = law_t(xt[idx], **ps.values()).squeeze() + std.sd_y * corr
        data = torch.mean(((pred - yt[idx]) / std.sd_y) ** 2)
        return data, torch.mean(corr**2)

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

    weight = float(w_phys)
    best = None  # (val_loss, epoch, state)
    stopped_at = epochs
    for ep in range(epochs):
        opt.zero_grad()
        data, phys = losses(fit_idx)
        if options.balance and weight > 0 and ep % options.balance_every == 0:
            weight = _annealed_weight(net, data, phys, weight, options.balance_alpha)
        (data + weight * phys).backward()
        opt.step()
        sched.step()

        if use_early and ep % options.check_every == 0:
            with torch.no_grad():
                v = float(losses(val_idx)[0])
            if best is None or v < best[0] - 1e-12:
                best = (v, ep, _snapshot(net, ps))
            elif ep - best[1] >= options.patience:
                stopped_at = ep
                break

        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["data_loss"].append(float(data.detach()))
            history["phys_loss"].append(float(phys.detach()))
            history["loss"].append(float((data + weight * phys).detach()))
            for k, v_ in ps.numpy().items():
                history[k].append(v_)
            if grid_np is not None:
                with torch.no_grad():
                    c = net(torch.tensor(std.x(grid_np), dtype=DTYPE)).squeeze(-1)
                    pq = law_t(
                        torch.tensor(grid_np, dtype=DTYPE), **ps.values()
                    ).squeeze()
                    history["pred"].append((pq + std.sd_y * c).numpy().ravel())

    if use_early and best is not None:
        _restore(net, ps, best[2])

    lbfgs_kept = None
    if options.lbfgs:
        lbfgs_kept = _refine_lbfgs(
            net, ps, losses, fit_idx, weight, options.lbfgs_steps
        )

    theta = ps.numpy()

    def predict(xq: np.ndarray) -> np.ndarray:
        xq2 = np.atleast_2d(np.asarray(xq, float))
        if xq2.shape[1] != xs.shape[1]:
            xq2 = xq2.T
        with torch.no_grad():
            c = net(torch.tensor(std.x(xq2), dtype=DTYPE)).squeeze(-1)
            p = law_t(torch.tensor(xq2, dtype=DTYPE), **ps.values()).squeeze()
            return (p + std.sd_y * c).numpy().ravel()

    with torch.no_grad():
        final_data, _ = losses(fit_idx)
    return Fit(
        name=name,
        predict=predict,
        params=theta,
        n_free=len(params) + sum(p.numel() for p in net.parameters()),
        seconds=time.time() - t0,
        history=history if record_every else None,
        extra={
            "w_phys": w_phys,
            "w_phys_final": weight,
            "options": options.tag,
            "early_stopped_at": stopped_at if use_early else None,
            "early_stopping_used": use_early,
            "lbfgs_kept": lbfgs_kept,
            # Did the switch that was turned on actually do anything? An
            # option that cannot engage on a track must be reported as
            # inapplicable, not as an option that failed to help.
            "engaged": _engaged(options, use_early, lbfgs_kept),
            "correction_rms_frac": float(
                torch.sqrt(torch.mean(net(xs).squeeze(-1) ** 2)).detach()
            ),
            "data_mse_std_units": float(final_data),
        },
    )


def _engaged(options: PinnOptions, use_early: bool, lbfgs_kept) -> bool:
    if options.early_stopping and not use_early:
        return False
    return not (options.lbfgs and not lbfgs_kept)


def _snapshot(net, ps):
    return (
        {k: v.detach().clone() for k, v in net.state_dict().items()},
        {k: v.detach().clone() for k, v in ps.state_dict().items()},
    )


def _restore(net, ps, snap) -> None:
    net.load_state_dict(snap[0])
    ps.load_state_dict(snap[1])


def _annealed_weight(net, data, phys, weight: float, alpha: float) -> float:
    """Wang et al. (2021) learning-rate annealing.

    The physics weight is set from the ratio of gradient norms so neither
    term is invisible to the optimiser, then smoothed. Returns the old
    weight unchanged if either gradient vanishes -- a zero denominator here
    would otherwise send the weight to infinity on the first flat step.
    """
    shared = [p for p in net.parameters() if p.requires_grad]
    gd = torch.autograd.grad(data, shared, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(phys, shared, retain_graph=True, allow_unused=True)
    dmax = max((float(g.abs().max()) for g in gd if g is not None), default=0.0)
    pmean = float(
        np.mean([float(g.abs().mean()) for g in gp if g is not None] or [0.0])
    )
    if not np.isfinite(dmax) or not np.isfinite(pmean) or pmean <= 0 or dmax <= 0:
        return weight
    return alpha * weight + (1.0 - alpha) * (dmax / pmean)


def _refine_lbfgs(net, ps, losses, idx, weight: float, steps: int) -> bool:
    """A second-order polish, reverted if it does not help.

    L-BFGS on a loss this stiff can diverge outright; the fit it would
    return then is worse than the one Adam already had, so the step is
    treated as a proposal and kept only on the evidence.
    """
    before = _snapshot(net, ps)
    with torch.no_grad():
        d0, p0 = losses(idx)
        start = float(d0 + weight * p0)
    opt = torch.optim.LBFGS(
        list(net.parameters()) + list(ps.parameters()),
        max_iter=steps,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        data, phys = losses(idx)
        loss = data + weight * phys
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, ValueError):
        _restore(net, ps, before)
        return False
    with torch.no_grad():
        d1, p1 = losses(idx)
        end = float(d1 + weight * p1)
    if not np.isfinite(end) or end > start:
        _restore(net, ps, before)
        return False
    return True


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
