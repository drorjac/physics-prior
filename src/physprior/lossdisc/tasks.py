"""The test problems: second-order ODEs on t in [0, T].

Every task is u'' + F(u, u') = 0 with u(0) = u0, u'(0) = v0. The stiffness
scale is k = (2 pi f)^2, f the linear frequency in cycles per unit time, so
`f` sets how many oscillations the window holds and with it how hard the
problem is for a PINN. The reference solution is `solve_ivp` (DOP853) at
rtol 1e-11.

Forward tasks (trial A) give the network only the equation and the initial
condition. Inverse tasks (trial B) give it noisy observations and the form of
the equation, and ask for k.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
from scipy.integrate import solve_ivp

KINDS = ("sho", "damped", "duffing", "pendulum", "vdp")


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    f: float  # cycles per unit time
    u0: float = 1.0
    v0: float = 0.0
    c: float = 0.0  # damping (damped), or mu (vdp)
    beta: float = 0.0  # cubic stiffness relative to k (duffing)
    T: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown task kind {self.kind!r}")

    @property
    def k(self) -> float:
        return float((2 * np.pi * self.f) ** 2)

    def force(self, u, v, k=None, lib=np):
        """F(u, u') for numpy or torch arrays. `k` overrides the true k."""
        k = self.k if k is None else k
        if self.kind == "sho":
            return k * u
        if self.kind == "damped":
            return self.c * v + k * u
        if self.kind == "duffing":
            return k * (u + self.beta * u**3)
        if self.kind == "pendulum":
            return k * lib.sin(u)
        return -self.c * (1 - u**2) * v + k * u  # vdp

    def solution(self, n: int = 2001) -> tuple[np.ndarray, np.ndarray]:
        return _solve(self, n)


@cache
def _solve(task: Task, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, task.T, n)
    sol = solve_ivp(
        lambda _t, y: [y[1], -task.force(y[0], y[1])],
        (0.0, task.T),
        [task.u0, task.v0],
        t_eval=t,
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"reference solve failed for {task.name}: {sol.message}")
    return t, sol.y[0]


# ---------------------------------------------------------------------------
# trial A: forward problems
# ---------------------------------------------------------------------------

# meta-training: the search and the baseline tuning see only these
FORWARD_TRAIN = (
    Task("sho_f2.5", "sho", 2.5),
    Task("damped_f2.5", "damped", 2.5, c=2.0),
    Task("duffing_f2", "duffing", 2.0, beta=1.0),
)

# held out: other frequencies, and two equations the search never saw
FORWARD_TEST = (
    Task("sho_f2", "sho", 2.0),
    Task("sho_f3", "sho", 3.0),
    Task("damped_f3", "damped", 3.0, c=3.0),
    Task("duffing_f2.5", "duffing", 2.5, beta=0.5),
    Task("pendulum_f2.5", "pendulum", 2.5, u0=2.0),
    Task("vdp_f2", "vdp", 2.0, u0=1.5, c=4.0),
)


# ---------------------------------------------------------------------------
# trial B: inverse problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inverse:
    task: Task
    n_obs: int = 30
    noise: float = 0.05  # relative to the sd of the clean signal
    k_init: float = 0.3  # the starting k as a fraction of the truth

    @property
    def name(self) -> str:
        return self.task.name

    def observations(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Noisy samples of u at uniformly random times. The noise depends on
        the seed, so tuning and reporting seeds see different data."""
        rng = np.random.default_rng(10_000 + seed)
        t_ref, u_ref = self.task.solution()
        t = np.sort(rng.uniform(0.0, self.task.T, self.n_obs))
        u = np.interp(t, t_ref, u_ref)
        return t, u + self.noise * np.std(u_ref) * rng.standard_normal(self.n_obs)


INVERSE_TRAIN = (
    Inverse(Task("sho_f3", "sho", 3.0)),
    Inverse(Task("damped_f3", "damped", 3.0, c=2.0)),
    Inverse(Task("duffing_f2.5", "duffing", 2.5, beta=1.0)),
)

INVERSE_TEST = (
    Inverse(Task("sho_f2.5", "sho", 2.5)),
    Inverse(Task("sho_f4", "sho", 4.0)),
    Inverse(Task("damped_f3.5", "damped", 3.5, c=3.0)),
    Inverse(Task("duffing_f3", "duffing", 3.0, beta=0.5)),
    Inverse(Task("pendulum_f3", "pendulum", 3.0, u0=2.0)),
    Inverse(Task("vdp_f2.5", "vdp", 2.5, u0=1.5, c=3.0)),
)


def as_tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
