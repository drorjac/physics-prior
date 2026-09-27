"""Reconstruction methods, from most to least physics.

`physics`  Green's function in the loop. Unknowns: K source positions and
           strengths plus the boundary lift. Initialised by greedy matching
           pursuit over a grid of candidate source positions (the field is
           linear in strengths and lift), then refined by bounded nonlinear
           least squares with analytic Jacobians, from that start and from
           random starts; the lowest cost wins.
`pinn`     an MLP u(x) trained on data + the PDE residual at collocation
           points, with the K source positions and strengths as trainable
           parameters. It knows the PDE and the source shape, but must find
           the sources by gradient descent rather than through the Green's
           function.
`pigp`     the physics fit plus a Gaussian process on its residual: the
           learned correction for an incomplete model.
`gp`       Gaussian process, isotropic RBF, constant mean, ML-II.
`interp`   thin-plate-spline radial basis interpolation (smoothing tuned).
`nn`       black-box MLP (`physprior.methods.neural.train_mlp`), tuned on
           the tuning seeds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize

from physprior.methods.base import Fit
from physprior.units import require

from .fields import N_SOURCES, SOURCE_REGION, Geometry

METHODS = ("physics", "pinn", "pigp", "gp", "interp", "nn")


# ---------------------------------------------------------------------------
# physics: Green's function in the loop

CANDIDATES_PER_AXIS = {1: 41, 2: 13, 3: 7}
RANDOM_STARTS = 4


def _pack(c, q, a):
    return np.concatenate([c.ravel(), q, a])


def _unpack(theta, K, d):
    c = theta[: K * d].reshape(K, d)
    q = theta[K * d : K * d + K]
    a = theta[K * d + K :]
    return c, q, a


def _linear_solve(B: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return coef


def _greedy_start(geom: Geometry, x, y, K) -> np.ndarray:
    """Orthogonal matching pursuit over candidate source positions."""
    Bc, pos = geom.candidates(x, CANDIDATES_PER_AXIS[geom.d])
    H = geom.lift(x)
    chosen: list[int] = []
    for _ in range(K):
        cols = np.hstack([H, Bc[:, chosen]]) if chosen else H
        Q, _ = np.linalg.qr(cols)
        r = y - Q @ (Q.T @ y)
        Bp = Bc - Q @ (Q.T @ Bc)
        nrm = np.sum(Bp**2, axis=0)
        score = (Bp.T @ r) ** 2 / np.where(nrm > 1e-14, nrm, np.inf)
        score[chosen] = -1.0
        chosen.append(int(np.argmax(score)))
    return pos[chosen]


def _design(geom: Geometry, x, c) -> np.ndarray:
    cols = [geom.green(x, ck)[0] for ck in c]
    return np.column_stack([*cols, geom.lift(x)])


@dataclass
class PhysicsResult:
    centers: np.ndarray
    q: np.ndarray
    lift: np.ndarray
    cost: float
    at_bound: bool
    starts_agreeing: int


def fit_physics(
    x: np.ndarray,
    y: np.ndarray,
    geom: Geometry,
    *,
    K: int = N_SOURCES,
    seed: int = 0,
    random_starts: int = RANDOM_STARTS,
) -> Fit:
    t0 = time.time()
    x = np.atleast_2d(np.asarray(x, float))
    y = np.asarray(y, float).ravel()
    d = geom.d
    require(x.shape[1] == d, "physics fit: sensor dimension mismatch")
    lo, hi = geom.fit_region
    L = geom.n_lift

    def resid(theta):
        c, q, a = _unpack(theta, K, d)
        u = geom.lift(x) @ a
        for ck, qk in zip(c, q, strict=True):
            u = u + qk * geom.green(x, ck)[0]
        return u - y

    def jac(theta):
        c, q, _ = _unpack(theta, K, d)
        J = np.empty((len(x), K * d + K + L))
        for k in range(K):
            val, g = geom.green(x, c[k], grad=True)
            J[:, k * d : (k + 1) * d] = q[k] * g
            J[:, K * d + k] = val
        J[:, K * d + K :] = geom.lift(x)
        return J

    rng = np.random.default_rng([seed, d, 99])
    starts = [_greedy_start(geom, x, y, K)]
    for _ in range(random_starts):
        starts.append(rng.uniform(*SOURCE_REGION, size=(K, d)))
    blo = np.concatenate([np.full(K * d, lo), np.full(K + L, -np.inf)])
    bhi = np.concatenate([np.full(K * d, hi), np.full(K + L, np.inf)])
    runs = []
    for c0 in starts:
        coef = _linear_solve(_design(geom, x, c0), y)
        theta0 = _pack(c0, coef[:K], coef[K:])
        try:
            res = least_squares(
                resid,
                theta0,
                jac=jac,
                bounds=(blo, bhi),
                x_scale="jac",
                max_nfev=200,
            )
            runs.append(res)
        except (np.linalg.LinAlgError, ValueError):
            continue
    require(bool(runs), "physics fit: every start failed")
    best = min(runs, key=lambda r: r.cost)
    c, q, a = _unpack(best.x, K, d)
    span = hi - lo
    at_bound = bool(np.any(np.minimum(c - lo, hi - c) < 1e-3 * span))
    agree = int(sum(r.cost <= best.cost * 1.01 + 1e-15 for r in runs))

    def predict(xq):
        xq = np.atleast_2d(np.asarray(xq, float))
        u = geom.lift(xq) @ a
        for ck, qk in zip(c, q, strict=True):
            u = u + qk * geom.green(xq, ck)[0]
        return u

    params = {f"q{k}": float(q[k]) for k in range(K)}
    params.update({f"a{j}": float(a[j]) for j in range(L)})
    return Fit(
        name="physics",
        predict=predict,
        params=params,
        n_free=K * (d + 1) + L,
        seconds=time.time() - t0,
        extra={
            "centers": c,
            "q": q,
            "lift": a,
            "at_bound": at_bound,
            "starts": len(runs),
            "starts_agreeing": agree,
            "cost": float(best.cost),
        },
    )


def location_error(true_c: np.ndarray, est_c: np.ndarray) -> float:
    """Mean distance between true and recovered sources, optimally matched."""
    from scipy.optimize import linear_sum_assignment

    D = np.linalg.norm(true_c[:, None, :] - est_c[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    return float(np.mean(D[r, c]))


# ---------------------------------------------------------------------------
# Gaussian process, isotropic RBF, ML-II

GP_BOUNDS = {"l": (0.01, 3.0), "s_f": (1e-3, 10.0), "s_n": (1e-4, 2.0)}
GP_STARTS = ({"l": 0.1, "s_f": 1.0, "s_n": 0.1}, {"l": 0.4, "s_f": 1.0, "s_n": 0.05})
GP_MLII_MAX = 768  # hyperparameters are learned on at most this many sensors


def _sqdist(a, b):
    return np.maximum(
        np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T, 0.0
    )


def _gp_nll(v, D2, y):
    ell, sf, sn = np.exp(v)
    Kf = sf**2 * np.exp(-0.5 * D2 / ell**2)
    K = Kf + (sn**2 + 1e-8) * np.eye(len(y))
    try:
        c = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12, np.zeros(3)
    alpha = cho_solve(c, y)
    nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(c[0])))
    Ki = cho_solve(c, np.eye(len(y)))
    A = Ki - np.outer(alpha, alpha)
    dK = (Kf * D2 / ell**2, 2 * Kf, 2 * sn**2 * np.eye(len(y)))
    grad = np.array([0.5 * np.sum(A * m) for m in dK])
    return float(nll), grad


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mean=None,
    name: str = "gp",
    seed: int = 0,
) -> Fit:
    """GP regression. `mean(x)` is a fixed mean function (the physics fit for
    `pigp`); the GP models what it leaves, around a constant."""
    t0 = time.time()
    x = np.atleast_2d(np.asarray(x, float))
    y = np.asarray(y, float).ravel()
    m0 = mean(x) if mean is not None else np.zeros(len(y))
    r = y - m0
    mu = float(np.mean(r))
    sd = float(np.std(r)) or 1.0
    z = (r - mu) / sd
    idx = np.arange(len(y))
    if len(y) > GP_MLII_MAX:
        idx = np.random.default_rng([seed, 5]).choice(
            len(y), GP_MLII_MAX, replace=False
        )
    D2 = _sqdist(x[idx], x[idx])
    names = tuple(GP_BOUNDS)
    bnds = [tuple(np.log(GP_BOUNDS[k])) for k in names]
    runs = []
    for s in GP_STARTS:
        v0 = np.log([s[k] for k in names])
        runs.append(
            minimize(
                _gp_nll,
                v0,
                args=(D2, z[idx]),
                jac=True,
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 200},
            )
        )
    best = min(runs, key=lambda o: o.fun)
    ell, sf, sn = np.exp(best.x)
    Kxx = sf**2 * np.exp(-0.5 * _sqdist(x, x) / ell**2) + (sn**2 + 1e-8) * np.eye(
        len(y)
    )
    c = cho_factor(Kxx, lower=True)
    alpha = cho_solve(c, z)
    at_bound = {
        k: bool(min(abs(best.x[i] - bnds[i][0]), abs(best.x[i] - bnds[i][1])) < 0.01)
        for i, k in enumerate(names)
    }

    def predict(xq):
        xq = np.atleast_2d(np.asarray(xq, float))
        out = np.empty(len(xq))
        for s in range(0, len(xq), 4096):
            Ks = sf**2 * np.exp(-0.5 * _sqdist(xq[s : s + 4096], x) / ell**2)
            out[s : s + 4096] = Ks @ alpha
        base = mean(xq) if mean is not None else 0.0
        return base + mu + sd * out

    return Fit(
        name=name,
        predict=predict,
        n_free=3,
        seconds=time.time() - t0,
        extra={
            "hyper": {"l": float(ell), "s_f": float(sf * sd), "s_n": float(sn * sd)},
            "at_bound": at_bound,
        },
    )


# ---------------------------------------------------------------------------
# classical interpolation

INTERP_SMOOTHING = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


def fit_interp(x, y, *, smoothing: float = 0.0) -> Fit:
    from scipy.interpolate import RBFInterpolator

    t0 = time.time()
    x = np.atleast_2d(np.asarray(x, float))
    y = np.asarray(y, float).ravel()
    mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
    degree = 1 if len(y) > x.shape[1] + 1 else 0
    f = RBFInterpolator(
        x, (y - mu) / sd, kernel="thin_plate_spline", smoothing=smoothing, degree=degree
    )

    def predict(xq):
        return mu + sd * f(np.atleast_2d(np.asarray(xq, float)))

    return Fit(
        name="interp",
        predict=predict,
        seconds=time.time() - t0,
        extra={"smoothing": smoothing},
    )


# ---------------------------------------------------------------------------
# black-box MLP


def fit_nn(x, y, *, cfg: dict, seed: int, epochs: int) -> Fit:
    from physprior.methods.neural import train_mlp

    keys = ("width", "depth", "weight_decay")
    return train_mlp(x, y, seed=seed, epochs=epochs, **{k: cfg[k] for k in keys})


# ---------------------------------------------------------------------------
# PINN with trainable sources

PINN_W_GRID = (1e-3, 1e-2, 1e-1)
PINN_EPOCHS = {1: 3000, 2: 2500, 3: 1500}
PINN_COLLOCATION = {1: 256, 2: 512, 3: 768}


def fit_pinn(
    x: np.ndarray,
    y: np.ndarray,
    geom: Geometry,
    *,
    w_pde: float,
    seed: int,
    K: int = N_SOURCES,
    epochs: int | None = None,
    width: int = 48,
    depth: int = 3,
    lr: float = 3e-3,
) -> Fit:
    """Loss = MSE(u(x_s), y)/var(y) + w_pde mean(r^2)/f0^2, with
    r = -laplacian(u) - sum_k q_k g_w(x - c_k) at fresh uniform collocation
    points each epoch, and f0 the peak of a unit source."""
    import torch

    from physprior.methods.base import set_seed
    from physprior.methods.neural import DTYPE, mlp

    t0 = time.time()
    set_seed(seed)
    d = geom.d
    epochs = epochs or PINN_EPOCHS[d]
    x = np.atleast_2d(np.asarray(x, float))
    y = np.asarray(y, float).ravel()
    mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
    lo, hi = geom.eval_region
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    net = mlp(d, width, depth)
    # deterministic spread start for the sources: a Halton-like lattice
    gold = np.array([0.618034, 0.754878, 0.569840])[:d]
    c0 = 0.3 + 0.4 * ((np.arange(1, K + 1)[:, None] * gold[None]) % 1.0)
    c_t = torch.nn.Parameter(torch.tensor(c0, dtype=DTYPE))
    q_t = torch.nn.Parameter(torch.zeros(K, dtype=DTYPE))
    w = geom.w
    f0 = 1.0 / (2 * np.pi * w * w) ** (d / 2)
    xs = torch.tensor(x, dtype=DTYPE)
    ys = torch.tensor((y - mu) / sd, dtype=DTYPE)
    gen = torch.Generator().manual_seed(seed)

    def u_of(z):
        return mu + sd * net((z - centre) / half).squeeze(-1)

    def source(z):
        out = torch.zeros(z.shape[0], dtype=DTYPE)
        for k in range(K):
            if geom.case == "box":
                g = torch.ones(z.shape[0], dtype=DTYPE)
                for i in range(d):
                    a = torch.exp(-0.5 * ((z[:, i] - c_t[k, i]) / w) ** 2)
                    a = a - torch.exp(-0.5 * ((z[:, i] + c_t[k, i]) / w) ** 2)
                    a = a - torch.exp(-0.5 * ((z[:, i] + c_t[k, i] - 2) / w) ** 2)
                    g = g * a / np.sqrt(2 * np.pi * w * w)
            else:
                r2 = torch.sum((z - c_t[k]) ** 2, dim=1)
                g = torch.exp(-0.5 * r2 / w**2) / (2 * np.pi * w * w) ** (d / 2)
            out = out + q_t[k] * g
        return out

    params = [*net.parameters(), c_t, q_t]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_col = PINN_COLLOCATION[d]
    for _ in range(epochs):
        opt.zero_grad()
        z = lo + (hi - lo) * torch.rand(n_col, d, dtype=DTYPE, generator=gen)
        z.requires_grad_(True)
        u = u_of(z)
        gu = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        lap = torch.zeros_like(u)
        for i in range(d):
            lap = (
                lap + torch.autograd.grad(gu[:, i].sum(), z, create_graph=True)[0][:, i]
            )
        r = (-lap - source(z)) / f0
        data = torch.mean(((u_of(xs) - mu) / sd - ys) ** 2)
        loss = data + w_pde * torch.mean(r**2)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            c_t.clamp_(*geom.fit_region)

    def predict(xq):
        with torch.no_grad():
            return u_of(torch.tensor(np.atleast_2d(xq), dtype=DTYPE)).numpy()

    return Fit(
        name="pinn",
        predict=predict,
        params={f"q{k}": float(q_t[k].detach()) for k in range(K)},
        seconds=time.time() - t0,
        extra={
            "centers": c_t.detach().numpy().copy(),
            "w_pde": w_pde,
            "final_data_loss": float(data.detach()),
            "final_pde_loss": float(torch.mean(r.detach() ** 2)),
        },
    )
