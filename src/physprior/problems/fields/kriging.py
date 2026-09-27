"""Gaussian-process regression (kriging), in numpy and scipy.

The geostatistical baseline a meteorologist would reach for first. Two
variants share one implementation and differ only in the mean function:

  ordinary kriging    constant mean + GP. A black box with a smoothness
                      prior: it knows nothing about height.
  universal kriging   the lapse-rate law as the mean,
                          m(x) = T0 + a dlon + b dlat + Gamma z,
                      + GP on what the law leaves. This is the classical
                      "kriging with external drift" of station
                      interpolation (e.g. Hengl 2009, A Practical Guide to
                      Geostatistical Mapping, ch. 2).

Kernel: an anisotropic squared exponential in (east km, north km, height km),

    k(x, x') = s_f^2 exp(-|dh|^2 / 2 l_h^2 - dz^2 / 2 l_z^2) + s_n^2 delta,

one horizontal and one vertical length scale, because a kilometre up is not
a kilometre across. The mean coefficients are profiled out by generalised
least squares and the four hyperparameters chosen by maximising the
restricted (REML) marginal likelihood on the training stations only.

The likelihood is maximised from several fixed starting points. A
hyperparameter that ends within 1% (in log) of its bound is flagged: it has
not converged, whatever the optimiser reports.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from physprior.methods.base import Fit
from physprior.units import require

# km per degree on the WGS84 ellipsoid at mid-latitude, used only to turn
# lon/lat differences into an isotropic horizontal distance for the kernel.
# (Meridional degree 111.13 km at 45-47 N; zonal degree 111.32 cos(lat).)
KM_PER_DEG_LAT = 111.13
KM_PER_DEG_LON_EQ = 111.32

# (low, high) in physical units, for the log-parameters
BOUNDS = {
    "l_h_km": (5.0, 1000.0),
    "l_z_km": (0.05, 10.0),
    "s_f": (1e-3, 20.0),  # K
    "s_n": (1e-3, 10.0),  # K
}
STARTS = (
    {"l_h_km": 50.0, "l_z_km": 0.5, "s_f": 1.0, "s_n": 0.5},
    {"l_h_km": 150.0, "l_z_km": 1.5, "s_f": 3.0, "s_n": 1.0},
    {"l_h_km": 20.0, "l_z_km": 0.2, "s_f": 0.5, "s_n": 0.2},
)
NAMES = tuple(BOUNDS)


def to_km(x: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """(z_km, lon, lat) -> (east km, north km, z km)."""
    x = np.asarray(x, float)
    east = (x[:, 1] - lon0) * KM_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    north = (x[:, 2] - lat0) * KM_PER_DEG_LAT
    return np.column_stack([east, north, x[:, 0]])


def kernel(a: np.ndarray, b: np.ndarray, hp: dict[str, float]) -> np.ndarray:
    dh2 = (a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2
    dz2 = (a[:, None, 2] - b[None, :, 2]) ** 2
    return hp["s_f"] ** 2 * np.exp(
        -0.5 * dh2 / hp["l_h_km"] ** 2 - 0.5 * dz2 / hp["l_z_km"] ** 2
    )


@dataclass
class GPState:
    hp: dict[str, float]
    beta: np.ndarray
    beta_cov: np.ndarray
    alpha: np.ndarray  # K^-1 (y - H beta)
    X: np.ndarray  # training inputs, km
    nll: float


def _unpack(v: np.ndarray) -> dict[str, float]:
    return {k: float(np.exp(t)) for k, t in zip(NAMES, v, strict=True)}


def _condition(X, y, H, hp) -> GPState:
    K = kernel(X, X, hp) + (hp["s_n"] ** 2 + 1e-10) * np.eye(len(y))
    c = cho_factor(K, lower=True)
    Ki_H = cho_solve(c, H)
    Ki_y = cho_solve(c, y)
    A = H.T @ Ki_H
    ca = cho_factor(A, lower=True)
    beta = cho_solve(ca, H.T @ Ki_y)
    r = y - H @ beta
    alpha = cho_solve(c, r)
    # restricted likelihood: the mean coefficients integrated out
    logdet_k = 2.0 * np.sum(np.log(np.diag(c[0])))
    logdet_a = 2.0 * np.sum(np.log(np.diag(ca[0])))
    nll = 0.5 * (r @ alpha + logdet_k + logdet_a)
    return GPState(hp, beta, cho_solve(ca, np.eye(len(beta))), alpha, X, float(nll))


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    basis: Callable[[np.ndarray], np.ndarray],
    *,
    lon0: float,
    lat0: float,
    name: str = "gp",
    beta_names: tuple[str, ...] = (),
) -> Fit:
    """ML-II (REML) GP with mean `basis(x) @ beta`. `x` is (z_km, lon, lat)."""
    t0 = time.time()
    x = np.asarray(x, float)
    y = np.asarray(y, float).ravel()
    require(x.ndim == 2 and x.shape[1] == 3, "GP: x must be (z_km, lon, lat)")
    require(bool(np.all(np.abs(x[:, 0]) < 9.0)), "GP: height must be in km")
    X = to_km(x, lon0, lat0)
    H = basis(x)
    lo = np.log([BOUNDS[k][0] for k in NAMES])
    hi = np.log([BOUNDS[k][1] for k in NAMES])

    def nll(v):
        try:
            return _condition(X, y, H, _unpack(v)).nll
        except np.linalg.LinAlgError:
            return 1e12

    runs = []
    for s in STARTS:
        v0 = np.log([s[k] for k in NAMES])
        res = minimize(
            nll, v0, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True))
        )
        runs.append(res)
    best = min(runs, key=lambda r: r.fun)
    state = _condition(X, y, H, _unpack(best.x))
    at_bound = {
        k: bool(
            min(abs(best.x[i] - lo[i]), abs(best.x[i] - hi[i])) < 0.01 * (hi[i] - lo[i])
        )
        for i, k in enumerate(NAMES)
    }
    # starts that reach the same optimum: a crude check the optimum is global
    agree = int(sum(abs(r.fun - best.fun) < 0.1 for r in runs))

    def predict(xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, float)
        Xq = to_km(xq, lon0, lat0)
        return basis(xq) @ state.beta + kernel(Xq, state.X, state.hp) @ state.alpha

    sig = np.sqrt(np.diag(state.beta_cov))
    params = {n: float(b) for n, b in zip(beta_names, state.beta, strict=False)}
    psig = {n: float(s) for n, s in zip(beta_names, sig, strict=False)}
    return Fit(
        name=name,
        predict=predict,
        params=params,
        param_sigma=psig,
        n_free=len(state.beta) + len(NAMES),
        seconds=time.time() - t0,
        expression=None,
        extra={
            "hyper": state.hp,
            "at_bound": at_bound,
            "converged": not any(at_bound.values()),
            "starts_agreeing": agree,
            "starts": len(STARTS),
            "nll": state.nll,
        },
    )
