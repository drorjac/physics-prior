"""Scores for the Lorenz study.

state    RMSE of the reconstructed trajectory on a 1001-point grid over the
         window, per component divided by that component's sd, then averaged:
         0 is perfect, 1 is as bad as predicting the mean.
deriv    the same for du/dt against the true vector field; it is what any
         route from a trajectory to a law has to get right.
theta    |theta_hat / theta - 1| per constant, and their mean.
vpt      valid prediction time: the forecast from the end of the window stays
         valid while its standardised error is below 0.4 (Pathak et al. 2018,
         the threshold of the dynamics study), in Lyapunov times 1/lambda_1.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from physprior.dynamics.metrics import LORENZ_THR

from .system import LAMBDA1, THETA_NAMES, THETA_TRUE, forecast_error, reference, rk4

HORIZON = 12.0  # forecast length, time units (about 11 Lyapunov times)
N_FORECAST = 1201


def nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    if not np.isfinite(pred).all():
        return float("inf")
    sd = truth.std(axis=0)
    return float((np.sqrt(((pred - truth) ** 2).mean(axis=0)) / sd).mean())


def theta_errors(theta: np.ndarray | None) -> dict[str, float]:
    if theta is None or not np.isfinite(theta).all():
        nan = float("nan")
        return {f"err_{k}": nan for k in THETA_NAMES} | {"err_theta": nan}
    rel = np.abs(np.asarray(theta) / THETA_TRUE - 1.0)
    out = {f"err_{k}": float(v) for k, v in zip(THETA_NAMES, rel, strict=True)}
    out["err_theta"] = float(rel.mean())
    return out


@lru_cache(maxsize=64)
def _future(u0: tuple, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(t_end, t_end + HORIZON, N_FORECAST)
    u = reference(np.array(u0), np.r_[0.0, t])[1:]
    return t, u


def future(obs) -> tuple[np.ndarray, np.ndarray]:
    """The true continuation past the window: (t, u) on the forecast grid."""
    return _future(tuple(obs.u0), obs.t_end)


def valid_time(err: np.ndarray, t: np.ndarray, t0: float) -> float:
    """Lyapunov times from t0 until err first reaches LORENZ_THR."""
    bad = ~(err < LORENZ_THR)
    if not bad.any():
        return float((t[-1] - t0) * LAMBDA1)
    return float((t[int(np.argmax(bad))] - t0) * LAMBDA1)


def forecast_law(obs, u_end: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Integrate the law with the fitted constants from the fitted end state,
    over the forecast grid. Inf where it blows up."""
    t, _ = future(obs)
    sol = rk4(u_end, HORIZON, theta, n_out=N_FORECAST)
    if sol is None:
        return np.full((len(t), 3), np.inf)
    return sol


def forecast_scores(obs, pred: np.ndarray) -> dict[str, float]:
    t, u = future(obs)
    err = forecast_error(pred, u, obs.sd)
    return {"vpt": valid_time(err, t, obs.t_end)}


def forecast_curve(obs, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, u = future(obs)
    return t - obs.t_end, forecast_error(pred, u, obs.sd)
