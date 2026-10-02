"""SINDy: sparse regression of dynamics (Brunton, Proctor & Kutz 2016).

Where genetic programming searches over trees, SINDy fixes a library of
candidate terms Theta(X) = [1, x, v, x^2, xv, ..., sin x, cos x] and solves

    dX/dt = Theta(X) Xi,    with Xi sparse,

one column per state variable. The sparsity is found by sequentially
thresholded least squares (STLSQ): solve, zero every coefficient below a
threshold, re-solve on the survivors, repeat. It is linear algebra, so it is
fast and deterministic; the price is that the law must be a sparse linear
combination of library terms -- the library plays the role the operator set
plays in symbolic regression.

Its weak point is the left-hand side. dX/dt is not measured; it is estimated
from the sampled trajectory, and a finite difference divides the measurement
noise by the step: the derivative's noise is ~ sigma / dt. `derivative`
offers central differences and a Savitzky-Golay smoothed derivative, and
`derivative_convergence` / `noise_amplification` measure both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from physprior.units import require

# Standard acceleration of gravity, g_n = 9.80665 m/s^2 [CGPM 1901, exact by
# definition]. The pendulum uses L = 1 m, so g/L = 9.80665 s^-2.
G_STANDARD = 9.80665

# ---------------------------------------------------------------------------
# systems with known right-hand sides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class System:
    name: str
    state: tuple[str, ...]
    x0: tuple[float, ...]
    trig: bool  # does the library need sin/cos?
    # true coefficients: {state_derivative: {library term: coefficient}}
    truth: dict[str, dict[str, float]]

    def rhs(self, t: float, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class DampedOscillator(System):
    omega: float = 2.0
    zeta: float = 0.1

    def rhs(self, t, s):
        x, v = s
        return np.array([v, -(self.omega**2) * x - 2 * self.zeta * self.omega * v])


@dataclass(frozen=True)
class Pendulum(System):
    g_over_l: float = G_STANDARD
    b: float = 0.2

    def rhs(self, t, s):
        th, om = s
        return np.array([om, -self.g_over_l * np.sin(th) - self.b * om])


def damped_oscillator(omega: float = 2.0, zeta: float = 0.1) -> DampedOscillator:
    return DampedOscillator(
        name="damped_oscillator",
        state=("x", "v"),
        x0=(1.0, 0.0),
        trig=False,
        truth={"x": {"v": 1.0}, "v": {"x": -(omega**2), "v": -2 * zeta * omega}},
        omega=omega,
        zeta=zeta,
    )


def pendulum(g_over_l: float = G_STANDARD, b: float = 0.2) -> Pendulum:
    # started at 2 rad, far outside the small-angle regime, so sin(theta)
    # cannot be replaced by theta
    return Pendulum(
        name="pendulum",
        state=("theta", "omega"),
        x0=(2.0, 0.0),
        trig=True,
        truth={
            "theta": {"omega": 1.0},
            "omega": {"sin(theta)": -g_over_l, "omega": -b},
        },
        g_over_l=g_over_l,
        b=b,
    )


def simulate(
    sys: System, t_end: float = 10.0, dt: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Reference trajectory: DOP853 at rtol 1e-12, so the integration error
    is far below any derivative error studied here."""
    from scipy.integrate import solve_ivp

    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    sol = solve_ivp(
        sys.rhs, (0.0, t[-1]), sys.x0, method="DOP853", t_eval=t, rtol=1e-12, atol=1e-12
    )
    require(sol.success, f"integration failed for {sys.name}")
    return t, sol.y.T


def add_noise(X: np.ndarray, sigma_rel: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with standard deviation sigma_rel * std(column)."""
    return X + sigma_rel * X.std(axis=0) * rng.standard_normal(X.shape)


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------


def derivative(
    X: np.ndarray, dt: float, method: str = "fd", window: int = 21, polyorder: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """(states used in the library, their time derivatives).

    fd       second-order central differences (one-sided at the ends); the
             states are used as measured.
    savgol   Savitzky-Golay: a local least-squares polynomial of degree
             `polyorder` over `window` samples, differentiated analytically.
             The same filter also smooths the states, since the library
             sees measurement noise too.
    """
    if method == "fd":
        return X, np.gradient(X, dt, axis=0, edge_order=2)
    if method == "savgol":
        from scipy.signal import savgol_filter

        require(
            window % 2 == 1 and window > polyorder,
            "savgol window must be odd and > polyorder",
        )
        Xs = savgol_filter(X, window, polyorder, deriv=0, axis=0, mode="interp")
        dX = savgol_filter(
            X, window, polyorder, deriv=1, delta=dt, axis=0, mode="interp"
        )
        return Xs, dX
    raise ValueError(f"unknown derivative method {method!r}")


def derivative_convergence(
    dts=(0.2, 0.1, 0.05, 0.025, 0.0125), window_time: float = 0.4, t_end: float = 10.0
) -> list[dict]:
    """Error of each derivative on a CLEAN signal vs step size.

    x(t) = sin(2t), exact derivative 2cos(2t). Central differences should
    show slope 2 in log(error) vs log(dt). Savitzky-Golay is run at a fixed
    window in TIME (so its sample count grows as dt falls): its error is then
    dominated by the polynomial's fit bias over the window, which does not
    shrink with dt -- the price of smoothing. Errors are on the interior
    (ends excluded) so edge stencils do not dominate.
    """
    rows = []
    for dt in dts:
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        x = np.sin(2 * t)[:, None]
        exact = 2 * np.cos(2 * t)
        inner = slice(len(t) // 10, -len(t) // 10)
        _, d_fd = derivative(x, dt, "fd")
        win = max(5, round(window_time / dt) | 1)
        row = {
            "dt": dt,
            "fd_max_err": float(np.max(np.abs(d_fd[inner, 0] - exact[inner]))),
            "savgol_window": win,
        }
        if win < len(t):
            _, d_sg = derivative(x, dt, "savgol", window=win)
            row["savgol_max_err"] = float(np.max(np.abs(d_sg[inner, 0] - exact[inner])))
        rows.append(row)
    for a, b in itertools.pairwise(rows):
        b["fd_observed_order"] = float(
            np.log(a["fd_max_err"] / b["fd_max_err"]) / np.log(a["dt"] / b["dt"])
        )
    return rows


def noise_amplification(
    sigmas=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1),
    dt: float = 0.01,
    window: int = 21,
    seeds=(11, 23, 42),
) -> list[dict]:
    """RMS derivative error vs measurement noise, at fixed dt.

    For central differences the noise part is sigma / (sqrt(2) dt): at
    dt = 0.01 the derivative is ~70x noisier than the signal. The ratio
    error/sigma is reported so that constant is visible.
    """
    t = np.arange(0.0, 10.0 + 0.5 * dt, dt)
    clean = np.sin(2 * t)[:, None]
    exact = 2 * np.cos(2 * t)
    inner = slice(len(t) // 10, -len(t) // 10)
    rows = []
    for s in sigmas:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            x = clean + s * rng.standard_normal(clean.shape)
            for method in ("fd", "savgol"):
                _, d = derivative(x, dt, method, window=window)
                e = float(np.sqrt(np.mean((d[inner, 0] - exact[inner]) ** 2)))
                rows.append(
                    {
                        "sigma": s,
                        "seed": seed,
                        "method": method,
                        "rms_err": e,
                        "err_over_sigma": e / s,
                        "fd_theory": s / (np.sqrt(2) * dt),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# library and STLSQ
# ---------------------------------------------------------------------------


def library(
    X: np.ndarray, names: tuple[str, ...], degree: int = 2, trig: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Polynomials in the states up to `degree`, plus sin/cos of each state."""
    from itertools import combinations_with_replacement

    cols = [np.ones(len(X))]
    labels = ["1"]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            cols.append(np.prod(X[:, combo], axis=1))
            labels.append("*".join(names[i] for i in combo))
    if trig:
        for i, n in enumerate(names):
            cols += [np.sin(X[:, i]), np.cos(X[:, i])]
            labels += [f"sin({n})", f"cos({n})"]
    return np.column_stack(cols), labels


def stlsq(
    Theta: np.ndarray,
    dX: np.ndarray,
    threshold: float,
    max_iter: int = 20,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Sequentially thresholded least squares.

    Columns are normalised to unit norm before solving so one threshold means
    the same for every term; coefficients are mapped back afterwards and the
    threshold is applied to those (physical) coefficients.
    """
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1.0
    A = Theta / norms
    n_terms = A.shape[1]
    Xi = np.zeros((n_terms, dX.shape[1]))
    for j in range(dX.shape[1]):
        active = np.ones(n_terms, bool)
        coef = np.zeros(n_terms)
        for _ in range(max_iter):
            As = A[:, active]
            sol = np.linalg.solve(
                As.T @ As + ridge * np.eye(As.shape[1]), As.T @ dX[:, j]
            )
            coef = np.zeros(n_terms)
            coef[active] = sol / norms[active]
            new_active = np.abs(coef) >= threshold
            if new_active.sum() == 0 or np.array_equal(new_active, active):
                active = new_active
                break
            active = new_active
        else:
            # Out of iterations with the support still changing: the last
            # solve was on the previous support, so re-solve on the final one.
            As = A[:, active]
            sol = np.linalg.solve(
                As.T @ As + ridge * np.eye(As.shape[1]), As.T @ dX[:, j]
            )
            coef = np.zeros(n_terms)
            coef[active] = sol / norms[active]
        coef[~active] = 0.0
        Xi[:, j] = coef
    return Xi


@dataclass
class SindyModel:
    coef: np.ndarray  # (n_terms, n_states)
    labels: list[str]
    state: tuple[str, ...]

    def equations(self, digits: int = 4) -> list[str]:
        out = []
        for j, s in enumerate(self.state):
            terms = [
                f"{self.coef[i, j]:+.{digits}g} {lab}"
                for i, lab in enumerate(self.labels)
                if self.coef[i, j] != 0
            ]
            out.append(f"d{s}/dt = " + (" ".join(terms) if terms else "0"))
        return out


def fit_sindy(
    t: np.ndarray,
    X: np.ndarray,
    state: tuple[str, ...],
    *,
    threshold: float = 0.1,
    method: str = "fd",
    window: int = 21,
    degree: int = 2,
    trig: bool = False,
    trim: float = 0.05,
) -> SindyModel:
    """Estimate derivatives, build the library, run STLSQ. The first and last
    `trim` of the samples are dropped, where both derivative estimates use
    one-sided stencils."""
    dt = float(t[1] - t[0])
    Xs, dX = derivative(X, dt, method, window=window)
    k = int(trim * len(t))
    sl = slice(k, len(t) - k)
    Theta, labels = library(Xs[sl], state, degree, trig)
    return SindyModel(stlsq(Theta, dX[sl], threshold), labels, state)


def score_model(model: SindyModel, sys: System) -> dict:
    """Support recovered exactly? Relative coefficient error on the truth."""
    true = np.zeros_like(model.coef)
    for j, s in enumerate(sys.state):
        for term, c in sys.truth[s].items():
            true[model.labels.index(term), j] = c
    support_ok = bool(np.array_equal(model.coef != 0, true != 0))
    err = float(np.linalg.norm(model.coef - true) / np.linalg.norm(true))
    n_extra = int(np.sum((model.coef != 0) & (true == 0)))
    n_missing = int(np.sum((model.coef == 0) & (true != 0)))
    return {
        "support_ok": support_ok,
        "coef_rel_err": err,
        "n_extra": n_extra,
        "n_missing": n_missing,
    }
