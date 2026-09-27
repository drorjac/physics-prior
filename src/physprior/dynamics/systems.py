"""ODE systems with a known law, and the reference trajectories they produce.

Every system is written once as a vector field `f(u, ops)` that works on
numpy arrays and on torch tensors, so the reference integrator, the oracle
stepper and the known part of a closure model all evaluate the same code.

Units are dimensionless throughout: the pendulum has g/l = 1 and unit mass,
the Kepler problem has GM = 1 and semi-major axes near 1, so the orbital
period is 2*pi. Nothing here is a measured constant.

Reference trajectories are integrated with classical RK4 at `substeps` steps
per observation interval. `convergence()` is the study that justifies the
choice: it halves the step repeatedly, checks the observed order is four, and
reports the error of the step actually used against a finer one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import numpy as np

from physprior.numerics import integrators
from physprior.units import require

# Lorenz (1963) parameters, J. Atmos. Sci. 20, 130: sigma=10, rho=28, beta=8/3.
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
# Largest Lyapunov exponent of the Lorenz-63 attractor at those parameters,
# 0.9056 per unit time: J. C. Sprott, Chaos and Time-Series Analysis (Oxford
# University Press, 2003), Appendix A. `lyapunov_exponent()` re-measures it.
LORENZ_LAMBDA1 = 0.9056

# Duffing oscillator x'' + gamma x' + omega^2 x + beta x^3 = F cos(Omega t).
# Chosen, not measured; hardening spring and moderate drive, so the long-time
# response is a forced oscillation rather than chaos.
DUFFING = {"omega": 1.0, "gamma": 0.3, "beta": 1.0, "F": 0.5, "Omega": 1.2}

NP = SimpleNamespace(
    sin=np.sin, cos=np.cos, sqrt=np.sqrt, stack=lambda xs: np.stack(xs, axis=-1)
)


def torch_ops() -> SimpleNamespace:
    import torch

    return SimpleNamespace(
        sin=torch.sin,
        cos=torch.cos,
        sqrt=torch.sqrt,
        stack=lambda xs: torch.stack(xs, dim=-1),
    )


Field = Callable[[Any, SimpleNamespace], Any]


# ---------------------------------------------------------------------------
# vector fields


def pendulum_f(u, ops):
    q, p = u[..., 0], u[..., 1]
    return ops.stack([p, -ops.sin(q)])


def pendulum_known(u, ops):
    """Small-angle pendulum: the physics a closure model is given."""
    q, p = u[..., 0], u[..., 1]
    return ops.stack([p, -q])


def pendulum_H(u):
    return 0.5 * u[..., 1] ** 2 - np.cos(u[..., 0])


def duffing_f(u, ops):
    d = DUFFING
    x, v, c, s = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    a = -d["gamma"] * v - d["omega"] ** 2 * x - d["beta"] * x**3 + d["F"] * c
    return ops.stack([v, a, -d["Omega"] * s, d["Omega"] * c])


def duffing_known(u, ops):
    """Undamped linear oscillator with the known drive: damping and the cubic
    spring are what the closure has to learn."""
    d = DUFFING
    x, v, c, s = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    a = -(d["omega"] ** 2) * x + d["F"] * c
    return ops.stack([v, a, -d["Omega"] * s, d["Omega"] * c])


def kepler_f(u, ops):
    qx, qy, px, py = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    r3 = ops.sqrt(qx**2 + qy**2) ** 3
    return ops.stack([px, py, -qx / r3, -qy / r3])


def kepler_E(u):
    r = np.hypot(u[..., 0], u[..., 1])
    return 0.5 * (u[..., 2] ** 2 + u[..., 3] ** 2) - 1.0 / r


def kepler_L(u):
    return u[..., 0] * u[..., 3] - u[..., 1] * u[..., 2]


def lorenz_f(u, ops):
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return ops.stack(
        [
            LORENZ_SIGMA * (y - x),
            x * (LORENZ_RHO - z) - y,
            x * y - LORENZ_BETA * z,
        ]
    )


def lorenz_known(u, ops):
    """The linear part of Lorenz-63; the closure learns the two products."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return ops.stack([LORENZ_SIGMA * (y - x), LORENZ_RHO * x - y, -LORENZ_BETA * z])


# ---------------------------------------------------------------------------
# initial conditions


def _pendulum_ic(rng, n, split):
    lo, hi = (-0.9, 0.5) if split != "ood" else (0.55, 0.9)
    H = rng.uniform(lo, hi, n)
    amp = np.arccos(-H)  # turning point, H = -cos(amp)
    q = rng.uniform(-1, 1, n) * amp
    p = np.sqrt(np.maximum(2 * (H + np.cos(q)), 0.0)) * rng.choice([-1, 1], n)
    return np.stack([q, p], axis=-1)


def _duffing_ic(rng, n, split):
    if split == "ood":
        r = rng.uniform(1.6, 2.4, n)
    else:
        r = np.sqrt(rng.uniform(0, 1, n)) * 1.5
    th = rng.uniform(0, 2 * np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th), np.cos(ph), np.sin(ph)], -1)


def _kepler_ic(rng, n, split):
    a = rng.uniform(0.8, 1.2, n)
    e = rng.uniform(0.0, 0.5, n) if split != "ood" else rng.uniform(0.5, 0.6, n)
    rp = a * (1 - e)
    vp = np.sqrt((1 + e) / rp)  # vis-viva at perihelion, GM = 1
    w = rng.uniform(0, 2 * np.pi, n)
    q = rp[:, None] * np.stack([np.cos(w), np.sin(w)], -1)
    p = vp[:, None] * np.stack([-np.sin(w), np.cos(w)], -1)
    return np.concatenate([q, p], axis=-1)


def _lorenz_ic(rng, n, split):
    # Start near the attractor and let it settle so every IC is on it.
    u = rng.normal(0, 1, (n, 3)) * np.array([8.0, 8.0, 8.0]) + np.array([0, 0, 25.0])
    return rk4_first_order(lorenz_f, u, 0.01, 1500, stride=1500)[:, -1]


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OdeSystem:
    name: str
    dim: int
    dt: float  # observation interval, the step the model learns
    substeps: int  # reference RK4 steps per observation interval
    f: Field = field(repr=False)
    sample_ic: Callable[[np.random.Generator, int, str], np.ndarray] = field(repr=False)
    train_steps: int  # length of one training trajectory
    test_steps: int  # length of a test rollout
    known: Field | None = field(default=None, repr=False)
    # Canonical (q, p) split for a separable Hamiltonian H = T(p) + V(q).
    hamiltonian: bool = False
    invariants: dict[str, Callable[[np.ndarray], np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    chaotic: bool = False
    description: str = ""

    @property
    def n_q(self) -> int:
        return self.dim // 2

    def accel(self, q: np.ndarray) -> np.ndarray:
        """-dV/dq for the separable Hamiltonian systems (unit mass)."""
        u = np.concatenate([q, np.zeros_like(q)], axis=-1)
        return self.f(u, NP)[..., self.n_q :]


def get_system(name: str) -> OdeSystem:
    return SYSTEMS[name]()


SYSTEMS: dict[str, Callable[[], OdeSystem]] = {
    "pendulum": lambda: OdeSystem(
        "pendulum",
        2,
        0.1,
        10,
        pendulum_f,
        _pendulum_ic,
        200,
        1000,
        known=pendulum_known,
        hamiltonian=True,
        invariants={"H": pendulum_H},
        description="q'' = -sin q; H = p^2/2 - cos q",
    ),
    "duffing": lambda: OdeSystem(
        "duffing",
        4,
        0.1,
        10,
        duffing_f,
        _duffing_ic,
        200,
        1000,
        known=duffing_known,
        description="x'' + 0.3x' + x + x^3 = 0.5 cos(1.2 t), state (x, v, cos, sin)",
    ),
    "kepler": lambda: OdeSystem(
        "kepler",
        4,
        0.05,
        25,
        kepler_f,
        _kepler_ic,
        250,
        2000,
        hamiltonian=True,
        invariants={"E": kepler_E, "L": kepler_L},
        description="q'' = -q/|q|^3 (GM = 1); E and L conserved",
    ),
    "lorenz": lambda: OdeSystem(
        "lorenz",
        3,
        0.01,
        10,
        lorenz_f,
        _lorenz_ic,
        500,
        2000,
        known=lorenz_known,
        chaotic=True,
        description="Lorenz-63, sigma=10, rho=28, beta=8/3",
    ),
}


# ---------------------------------------------------------------------------
# reference integration


def rk4_first_order(
    f: Field, u0: np.ndarray, h: float, n_steps: int, stride: int = 1
) -> np.ndarray:
    """Classical RK4 for u' = f(u) on a batch (B, d). Returns (B, K, d) with
    every `stride`-th state, the initial one included."""
    u = np.array(u0, float)
    out = [u.copy()]
    for k in range(1, n_steps + 1):
        k1 = f(u, NP)
        k2 = f(u + 0.5 * h * k1, NP)
        k3 = f(u + 0.5 * h * k2, NP)
        k4 = f(u + h * k3, NP)
        u = u + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % stride == 0:
            out.append(u.copy())
    return np.stack(out, axis=1)


def simulate(
    sys: OdeSystem, u0: np.ndarray, n_steps: int, substeps: int | None = None
) -> np.ndarray:
    """Reference trajectory sampled at the observation interval: (B, n+1, d)."""
    m = substeps or sys.substeps
    h = sys.dt / m
    u0 = np.atleast_2d(np.asarray(u0, float))
    if sys.hamiltonian:
        # Second-order form, reusing the project's integrator; p = v (unit mass).
        nq = sys.n_q
        _, R, V = integrators.rk4(
            sys.accel, u0[:, :nq], u0[:, nq:], h, n_steps * m, stride=m
        )
        out = np.concatenate([R, V], axis=-1).transpose(1, 0, 2)
    else:
        out = rk4_first_order(sys.f, u0, h, n_steps * m, stride=m)
    require(
        bool(np.all(np.isfinite(out))), f"{sys.name}: reference trajectory not finite"
    )
    return out


def convergence(sys: OdeSystem, n_ics: int = 4, seed: int = 0) -> dict:
    """Step-halving study of the reference integrator over one training
    trajectory. Errors are measured against a run with 4x the used substeps,
    in units of the state's standard deviation."""
    rng = np.random.default_rng(seed)
    u0 = sys.sample_ic(rng, n_ics, "test")
    n = sys.train_steps
    fine = simulate(sys, u0, n, sys.substeps * 4)
    scale = fine.reshape(-1, sys.dim).std(axis=0)
    rows = []
    for m in (sys.substeps // 4, sys.substeps // 2, sys.substeps, sys.substeps * 2):
        m = max(m, 1)
        tr = simulate(sys, u0, n, m)
        err = float(np.max(np.abs((tr - fine) / scale)))
        rows.append({"substeps": m, "h": sys.dt / m, "max_err": err})
    for a, b in pairwise(rows):
        b["observed_order"] = (
            float(np.log2(a["max_err"] / b["max_err"]))
            if a["max_err"] > 0 and b["max_err"] > 0
            else float("nan")
        )
    used = next(r for r in rows if r["substeps"] == sys.substeps)
    return {
        "system": sys.name,
        "rows": rows,
        "used_substeps": sys.substeps,
        "used_max_err": used["max_err"],
        "horizon_steps": n,
    }


def lyapunov_exponent(t_total: float = 200.0, seed: int = 0) -> float:
    """Largest Lyapunov exponent of Lorenz-63 by two-trajectory
    renormalisation (Benettin et al. 1980), on the reference integrator."""
    sys = get_system("lorenz")
    rng = np.random.default_rng(seed)
    u = sys.sample_ic(rng, 1, "test")
    d0 = 1e-8
    w = u + d0 * np.array([[1.0, 0.0, 0.0]])
    n = round(t_total / sys.dt)
    h = sys.dt / sys.substeps
    acc = 0.0
    for _ in range(n):
        u = rk4_first_order(sys.f, u, h, sys.substeps, stride=sys.substeps)[:, -1]
        w = rk4_first_order(sys.f, w, h, sys.substeps, stride=sys.substeps)[:, -1]
        d = float(np.linalg.norm(w - u))
        acc += np.log(d / d0)
        w = u + (w - u) * (d0 / d)
    return acc / (n * sys.dt)


# ---------------------------------------------------------------------------
# data sets


@dataclass
class OdeData:
    system: str
    train: np.ndarray  # (n_traj, train_steps+1, d)
    val: np.ndarray
    test: np.ndarray  # (n_test, test_steps+1, d), same distribution as train
    ood: np.ndarray | None  # (n_test, test_steps+1, d), unseen region
    mean: np.ndarray
    std: np.ndarray
    dstd: np.ndarray  # std of (u_{n+1}-u_n)/dt, the scale of a vector field


TEST_SEED, VAL_SEED = 1000, 999


def fixed_sets(
    sys: OdeSystem, n_test: int = 16, n_val: int = 8, n_steps: int | None = None
) -> dict[str, np.ndarray | None]:
    """Validation, test and out-of-distribution rollouts. They do not depend
    on the seed, so every arm and every seed is scored on the same states."""
    n = n_steps or sys.test_steps
    val = simulate(sys, sys.sample_ic(np.random.default_rng(VAL_SEED), n_val, "val"), n)
    trng = np.random.default_rng(TEST_SEED)
    test = simulate(sys, sys.sample_ic(trng, n_test, "test"), n)
    ood = None if sys.chaotic else simulate(sys, sys.sample_ic(trng, n_test, "ood"), n)
    return {"val": val, "test": test, "ood": ood}


def train_set(sys: OdeSystem, n_traj: int, seed: int) -> np.ndarray:
    """Training trajectories; the initial states depend on the seed."""
    rng = np.random.default_rng(seed)
    return simulate(sys, sys.sample_ic(rng, n_traj, "train"), sys.train_steps)


def assemble(sys: OdeSystem, train: np.ndarray, fixed: dict) -> OdeData:
    flat = train.reshape(-1, sys.dim)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    inc = ((train[:, 1:] - train[:, :-1]) / sys.dt).reshape(-1, sys.dim)
    dstd = inc.std(axis=0)
    dstd[dstd == 0] = 1.0
    return OdeData(
        sys.name,
        train,
        fixed["val"],
        fixed["test"],
        fixed["ood"],
        flat.mean(axis=0),
        std,
        dstd,
    )


def make_data(
    sys: OdeSystem, n_traj: int, seed: int, n_test: int = 16, n_val: int = 8
) -> OdeData:
    return assemble(sys, train_set(sys, n_traj, seed), fixed_sets(sys, n_test, n_val))
