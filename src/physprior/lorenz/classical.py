"""Fits of the Lorenz constants that do not train a network.

`regress_theta`    least squares for (sigma, rho, beta) given states and their
                   time derivatives. Each equation of the law is linear in one
                   constant, so this is three one-parameter regressions. It is
                   how a black-box network is turned into constants: fit the
                   trajectory, differentiate it, regress.
`single_shooting`  the classical inverse problem: integrate the law from a
                   guessed state and constants, compare with the observations,
                   let Levenberg-Marquardt-type least squares move all six.
`multiple_shooting` the standard remedy for chaotic and unstable systems
                   (Bock 1981; Baake et al. 1992): the window is cut into
                   segments, each with its own initial state, and continuity
                   between them is a residual rather than a constraint. Each
                   segment is short compared with the Lyapunov time, so the
                   residual surface is far smoother.
`polish`           single shooting started from another method's answer.

Every shooting fit starts from the same wrong constants THETA_INIT as the
networks do, unless it is a polish.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from physprior.units import require

from .system import RK4_H, THETA_INIT, Observations, lorenz_rhs, rk4

PENALTY = 1e2  # residual value used when a trial solution blows up


def regress_theta(u: np.ndarray, du: np.ndarray) -> np.ndarray:
    """(sigma, rho, beta) from states u (n, 3) and derivatives du (n, 3)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    dx, dy, dz = du[:, 0], du[:, 1], du[:, 2]
    sigma = np.dot(y - x, dx) / np.dot(y - x, y - x)
    rho = np.dot(x, dy + y + x * z) / np.dot(x, x)
    beta = np.dot(z, x * y - dz) / np.dot(z, z)
    return np.array([sigma, rho, beta])


@dataclass
class ShootingFit:
    method: str
    theta: np.ndarray
    u0: np.ndarray
    cost: float  # 0.5 * sum of squared standardised data residuals
    nfev: int
    seconds: float

    def predict(self, t: np.ndarray) -> np.ndarray:
        return trajectory(self.u0, self.theta, t)


def trajectory(u0: np.ndarray, theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """RK4 solution at times t (sorted, from 0); NaN if it blows up."""
    t_end = float(t[-1])
    n = max(1, round(t_end / RK4_H))
    sol = rk4(u0, t_end, theta, h=t_end / n)
    if sol is None:
        return np.full((len(t), 3), np.nan)
    tg = np.linspace(0.0, t_end, n + 1)
    return np.stack([np.interp(t, tg, sol[:, k]) for k in range(3)], axis=-1)


def _theta(p: np.ndarray) -> np.ndarray:
    return THETA_INIT * np.exp(p[:3])


def _data_cost(obs: Observations, theta: np.ndarray, u0: np.ndarray) -> float:
    pred = trajectory(u0, theta, obs.t_obs)
    if not np.isfinite(pred).all():
        return float("inf")
    return float(0.5 * (((pred - obs.y_obs) / obs.sd) ** 2).sum())


def single_shooting(
    obs: Observations,
    theta0: np.ndarray = THETA_INIT,
    u0: np.ndarray | None = None,
    max_nfev: int = 300,
    method: str = "shooting",
) -> ShootingFit:
    """All six unknowns by least squares on the integrated trajectory. The
    initial state defaults to the first observation, the natural guess."""
    u0 = obs.y_obs[0] if u0 is None else np.asarray(u0, float)
    t = obs.t_obs

    def resid(p):
        pred = trajectory(p[3:], _theta(p), t)
        if not np.isfinite(pred).all():
            return np.full(t.size * 3, PENALTY)
        return ((pred - obs.y_obs) / obs.sd).ravel()

    p0 = np.r_[np.log(np.asarray(theta0) / THETA_INIT), u0]
    t0 = time.perf_counter()
    r = least_squares(resid, p0, method="trf", x_scale="jac", max_nfev=max_nfev)
    th = _theta(r.x)
    return ShootingFit(
        method,
        th,
        r.x[3:],
        _data_cost(obs, th, r.x[3:]),
        r.nfev,
        time.perf_counter() - t0,
    )


def multiple_shooting(
    obs: Observations,
    n_seg: int = 6,
    w_cont: float = 10.0,
    max_nfev: int = 300,
) -> ShootingFit:
    """Segments [tau_k, tau_k+1) with free initial states; continuity is a
    residual of weight `w_cont`. Node states start at the observation nearest
    each node, the constants at THETA_INIT."""
    require(n_seg >= 1, "need at least one segment")
    T = obs.t_end
    tau = np.linspace(0.0, T, n_seg + 1)
    seg = np.clip(np.searchsorted(tau, obs.t_obs, side="right") - 1, 0, n_seg - 1)
    near = [int(np.argmin(np.abs(obs.t_obs - tk))) for tk in tau[:-1]]
    L = T / n_seg
    n = max(1, round(L / RK4_H))
    tg = np.linspace(0.0, L, n + 1)

    def resid(p):
        th = _theta(p)
        nodes = p[3:].reshape(n_seg, 3)
        sol = rk4(nodes, L, th, h=L / n)  # (n+1, n_seg, 3)
        if sol is None:
            return np.full(obs.t_obs.size * 3 + (n_seg - 1) * 3, PENALTY)
        rd = []
        for k in range(n_seg):
            m = seg == k
            if m.any():
                dt = obs.t_obs[m] - tau[k]
                pred = np.stack(
                    [np.interp(dt, tg, sol[:, k, c]) for c in range(3)], axis=-1
                )
                rd.append(((pred - obs.y_obs[m]) / obs.sd).ravel())
        cont = (sol[-1, :-1] - nodes[1:]) / obs.sd
        return np.concatenate([*rd, w_cont * cont.ravel()])

    p0 = np.r_[np.zeros(3), obs.y_obs[near].ravel()]
    t0 = time.perf_counter()
    r = least_squares(resid, p0, method="trf", x_scale="jac", max_nfev=max_nfev)
    th = _theta(r.x)
    u0 = r.x[3:6]
    return ShootingFit(
        "multiple_shooting",
        th,
        u0,
        _data_cost(obs, th, u0),
        r.nfev,
        time.perf_counter() - t0,
    )


def polish(
    obs: Observations, theta: np.ndarray, u0: np.ndarray, method: str
) -> ShootingFit:
    """Single shooting started from another method's constants and state."""
    return single_shooting(obs, theta, u0, method=method)


def fd_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second-order finite differences on uneven times (numpy.gradient): the
    route to constants with no model at all."""
    return np.gradient(y, t, axis=0, edge_order=2)


def residual_norm(u: np.ndarray, du: np.ndarray, theta: np.ndarray) -> float:
    return float(np.sqrt(((du - lorenz_rhs(u, theta)) ** 2).mean()))
