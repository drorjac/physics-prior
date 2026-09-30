"""Lorenz-63, the observations of it, and its sensitivity to initial state.

The law (Lorenz 1963, J. Atmos. Sci. 20, 130):

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

with sigma = 10, rho = 28, beta = 8/3 on the classical attractor.

The reference trajectory is integrated with DOP853 at rtol = atol = 1e-12.
Every other integration in the study (shooting, forecasts) uses the
fixed-step RK4 below, whose step is chosen by `rk4_convergence`: a result
that still moves with the step size is not a result.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from physprior.dynamics.systems import (
    LORENZ_BETA,
    LORENZ_LAMBDA1,
    LORENZ_RHO,
    LORENZ_SIGMA,
)
from physprior.exceptions import ConvergenceError
from physprior.units import require

THETA_TRUE = np.array([LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA])
THETA_NAMES = ("sigma", "rho", "beta")
# Every fitted method starts from the same wrong constants: each is off by a
# factor between 1.9 and 2.7, so none of them can be reached by a local
# correction of the truth.
THETA_INIT = np.array([5.0, 15.0, 1.0])
LAMBDA1 = LORENZ_LAMBDA1
# RK4 step for shooting and forecasts; see rk4_convergence().
RK4_H = 1e-3


def lorenz_rhs(u: np.ndarray, theta: np.ndarray = THETA_TRUE) -> np.ndarray:
    """The vector field on a batch (..., 3)."""
    s, r, b = theta
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return np.stack([s * (y - x), x * (r - z) - y, x * y - b * z], axis=-1)


def rk4(
    u0: np.ndarray,
    t_end: float,
    theta: np.ndarray = THETA_TRUE,
    h: float = RK4_H,
    n_out: int | None = None,
    bound: float = 1e4,
) -> np.ndarray | None:
    """Fixed-step RK4 from t = 0 to t_end, batched over leading axes of u0.

    Returns the states at `n_out` equally spaced times (all steps when None),
    shape (n_out, ..., 3), or None if the solution leaves |u| < bound -- a
    wrong theta can make the system blow up, and a shooting fit must see that
    as a large residual rather than wait on a stiff solve.
    """
    n = max(1, round(t_end / h))
    h = t_end / n
    keep = np.arange(n + 1) if n_out is None else np.linspace(0, n, n_out).round()
    keep_set = {int(k) for k in keep}
    u = np.array(u0, float)
    out = [u.copy()] if 0 in keep_set else []
    for k in range(1, n + 1):
        k1 = lorenz_rhs(u, theta)
        k2 = lorenz_rhs(u + 0.5 * h * k1, theta)
        k3 = lorenz_rhs(u + 0.5 * h * k2, theta)
        k4 = lorenz_rhs(u + h * k3, theta)
        u = u + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(u).all() or np.abs(u).max() > bound:
            return None
        if k in keep_set:
            out.append(u.copy())
    return np.stack(out)


def reference(u0: np.ndarray, t: np.ndarray, theta=THETA_TRUE) -> np.ndarray:
    """High-accuracy trajectory at times t (sorted, t[0] >= 0): (len(t), 3)."""
    sol = solve_ivp(
        lambda _t, u: lorenz_rhs(u, theta),
        (0.0, float(t[-1])),
        np.asarray(u0, float),
        t_eval=t,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    require(sol.success, f"reference integration failed: {sol.message}")
    return sol.y.T


def attractor_state(seed: int) -> np.ndarray:
    """A point on the attractor: a random start, run 20 time units forward."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 8.0, 3) + np.array([0.0, 0.0, 25.0])
    return reference(u, np.array([0.0, 20.0]))[-1]


@dataclass(frozen=True)
class Observations:
    """One trajectory, sampled sparsely and noisily, plus the truth behind it."""

    seed: int
    t_end: float
    noise: float  # noise sd as a fraction of each component's sd
    u0: np.ndarray  # true state at t = 0
    t_obs: np.ndarray  # (n,)
    y_obs: np.ndarray  # (n, 3), noisy
    t_dense: np.ndarray  # (m,), the evaluation grid over [0, t_end]
    u_dense: np.ndarray  # (m, 3), true
    sd: np.ndarray  # (3,), sd of the true trajectory on the window

    @property
    def n_obs(self) -> int:
        return len(self.t_obs)

    @property
    def du_dense(self) -> np.ndarray:
        return lorenz_rhs(self.u_dense)


def observe(
    seed: int,
    n_obs: int = 40,
    noise: float = 0.05,
    t_end: float = 3.0,
    n_dense: int = 1001,
) -> Observations:
    """`n_obs` observation times drawn uniformly on [0, t_end], Gaussian noise
    of sd `noise * sd_k` on component k. The seed fixes the trajectory, the
    times and the noise; the same seed gives the same trajectory at every
    noise level and budget, so the sweeps vary one thing at a time."""
    require(n_obs >= 2, "need at least two observations")
    require(noise >= 0.0, "noise is a standard deviation")
    u0 = attractor_state(seed)
    t_dense = np.linspace(0.0, t_end, n_dense)
    u_dense = reference(u0, t_dense)
    sd = u_dense.std(axis=0)
    rng = np.random.default_rng(seed + 1000)
    t_obs = np.sort(rng.uniform(0.0, t_end, n_obs))
    u_obs = reference(u0, t_obs)
    eps = rng.standard_normal(u_obs.shape)
    y_obs = u_obs + noise * sd * eps
    return Observations(seed, t_end, noise, u0, t_obs, y_obs, t_dense, u_dense, sd)


# ---------------------------------------------------------------------------
# chaos


def separation(
    u0: np.ndarray, delta: float = 1e-8, t_end: float = 30.0, n_out: int = 3001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two trajectories that start `delta` apart in x: (t, a, b)."""
    t = np.linspace(0.0, t_end, n_out)
    a = reference(u0, t)
    b = reference(u0 + np.array([delta, 0.0, 0.0]), t)
    return t, a, b


def growth_rate(t: np.ndarray, dist: np.ndarray, lo: float, hi: float) -> float:
    """Slope of log(distance) over the stretch where lo < distance < hi: the
    exponential phase, before the separation saturates at the attractor size."""
    m = (dist > lo) & (dist < hi)
    require(bool(m.sum() > 10), "too few points in the exponential phase")
    return float(np.polyfit(t[m], np.log(dist[m]), 1)[0])


def butterfly_ensemble(
    u0: np.ndarray, n: int = 24, delta: float = 1e-5, t_end: float = 25.0, seed=0
) -> tuple[np.ndarray, np.ndarray]:
    """`n` trajectories from a ball of radius `delta` around u0: (t, U) with
    U of shape (len(t), n, 3). Integrated together with RK4 at RK4_H."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d *= delta / np.linalg.norm(d, axis=1, keepdims=True)
    n_out = round(t_end / 0.01) + 1
    U = rk4(u0 + d, t_end, n_out=n_out)
    if U is None:
        raise ConvergenceError("the ensemble left the attractor")
    return np.linspace(0.0, t_end, n_out), U


def rk4_convergence(t_end: float = 3.0, seed: int = 11) -> list[dict]:
    """Error of RK4 against DOP853 at t_end for a sequence of steps; the
    observed order should be 4 and RK4_H should sit well below every error
    the study reports."""
    u0 = attractor_state(seed)
    ref = reference(u0, np.array([0.0, t_end]))[-1]
    rows = []
    for h in (4e-3, 2e-3, 1e-3, 5e-4):
        out = rk4(u0, t_end, h=h, n_out=2)
        if out is None:
            raise ConvergenceError("RK4 blew up on the true constants")
        rows.append({"h": h, "err": float(np.abs(out[-1] - ref).max())})
    for a, b in itertools.pairwise(rows):
        b["order"] = float(np.log2(a["err"] / b["err"]))
    return rows


def forecast_error(pred: np.ndarray, truth: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-time forecast error in standardised units (the dynamics study's
    convention): sqrt(mean_k ((pred_k - truth_k) / sd_k)^2)."""
    return np.sqrt((((pred - truth) / sd) ** 2).mean(axis=-1))
