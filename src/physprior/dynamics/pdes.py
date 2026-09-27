"""Periodic 1-D PDEs with a known law, on the grid the models learn on.

The learned steppers act on a coarse grid of N = 64 points on [0, 2 pi).

heat       u_t = kappa u_xx          exact in Fourier space
advection  u_t + c u_x = 0           exact in Fourier space
burgers    u_t + u u_x = nu u_xx     shock-forming; pseudo-spectral reference

Initial conditions are band-limited (Fourier modes up to K_MAX, below the
coarse grid's Nyquist), so for heat and advection the Fourier solution
sampled on the grid is exact and there is no discretisation to converge.
Burgers is solved on a fine grid (N_FINE points, 2/3 dealiasing, integrating
factor for the viscous term, RK4 in time) and then sampled at the coarse
points; `burgers_convergence()` refines both space and time to show the
sampled reference has stopped moving. On the coarse grid the Burgers fronts
are about one cell wide, so learning the coarse update includes learning a
sub-grid closure.

The `physics` stepper is the textbook scheme on the coarse grid: second-order
central differences, RK4 in time with enough substeps to be stable. It is
what the correct equation buys without learning anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from physprior.units import require

N_GRID = 64
L_DOMAIN = 2 * np.pi
N_FINE = 1024
K_MAX = {"train": 6, "test": 6, "val": 6, "ood": 10}

# Chosen, not measured.
KAPPA = 0.05  # heat diffusivity
C_ADV = 1.0  # advection speed
NU = 0.03  # Burgers viscosity; fronts ~ 2 nu / du wide, below the coarse cell


@dataclass(frozen=True)
class PdeSystem:
    name: str
    dt: float
    train_steps: int
    test_steps: int
    description: str
    physics_rhs: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    physics_substeps: int
    # Part of the RHS a closure stepper is given (None: no closure arm).
    known_rhs: Callable | None = field(default=None, repr=False)

    @property
    def dx(self) -> float:
        return L_DOMAIN / N_GRID


def grid(n: int = N_GRID) -> np.ndarray:
    return np.arange(n) * (L_DOMAIN / n)


def _ddx(u, dx, lib=np):
    return (lib.roll(u, -1, -1) - lib.roll(u, 1, -1)) / (2 * dx)


def _d2dx(u, dx, lib=np):
    return (lib.roll(u, -1, -1) - 2 * u + lib.roll(u, 1, -1)) / dx**2


def _roll_t(u, s, dim):
    import torch

    return torch.roll(u, s, dim)


class _TorchLib:
    @staticmethod
    def roll(u, s, dim):
        return _roll_t(u, s, dim)


def heat_rhs(u, lib=np):
    return KAPPA * _d2dx(u, L_DOMAIN / N_GRID, lib)


def adv_rhs(u, lib=np):
    return -C_ADV * _ddx(u, L_DOMAIN / N_GRID, lib)


def burgers_rhs(u, lib=np):
    dx = L_DOMAIN / N_GRID
    return -_ddx(0.5 * u * u, dx, lib) + NU * _d2dx(u, dx, lib)


def burgers_known(u):
    """Viscous term on the coarse grid (torch): what the closure is given."""
    return NU * _d2dx(u, L_DOMAIN / N_GRID, _TorchLib)


PDES: dict[str, PdeSystem] = {
    "heat": PdeSystem("heat", 0.05, 40, 200, "u_t = 0.05 u_xx", heat_rhs, 4),
    "advection": PdeSystem("advection", 0.05, 40, 400, "u_t + u_x = 0", adv_rhs, 2),
    "burgers": PdeSystem(
        "burgers",
        0.05,
        40,
        100,
        "u_t + u u_x = 0.03 u_xx",
        burgers_rhs,
        8,
        known_rhs=burgers_known,
    ),
}


# ---------------------------------------------------------------------------
# initial conditions and exact / reference solutions


def sample_ic(rng: np.random.Generator, n: int, split: str, name: str) -> np.ndarray:
    """Random band-limited fields, returned as Fourier coefficients on the
    coarse grid's rfft layout (length N_GRID // 2 + 1)."""
    kmax = K_MAX[split]
    k = np.arange(1, kmax + 1)
    amp = rng.normal(0, 1, (n, kmax)) / k ** (1.0 if split != "ood" else 0.5)
    ph = rng.uniform(0, 2 * np.pi, (n, kmax))
    coef = np.zeros((n, N_GRID // 2 + 1), complex)
    coef[:, 1 : kmax + 1] = amp * np.exp(1j * ph)
    # Normalise the fluctuation to a fixed RMS so amplitude is controlled.
    u = np.fft.irfft(coef, n=N_GRID, axis=-1)
    rms = u.std(axis=-1, keepdims=True)
    target = 0.5 if split != "ood" else 0.8
    coef *= target / rms
    if name == "burgers":
        # A mean flow makes the fronts travel, so position matters.
        coef[:, 0] = rng.uniform(-0.5, 0.5, n) * N_GRID
    return coef


def exact_linear(name: str, coef: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    k = np.arange(coef.shape[-1])
    t = np.arange(n_steps + 1) * dt
    if name == "heat":
        fac = np.exp(-KAPPA * k[None, :] ** 2 * t[:, None])
    else:
        fac = np.exp(-1j * C_ADV * k[None, :] * t[:, None])
    return np.fft.irfft(coef[:, None, :] * fac[None], n=N_GRID, axis=-1)


def burgers_reference(
    coef: np.ndarray, n_steps: int, dt: float, n_fine: int = N_FINE, sub: int = 25
) -> np.ndarray:
    """Integrating-factor RK4 pseudo-spectral solve on `n_fine` points, 2/3
    dealiased, sampled at the N_GRID coarse points every dt."""
    B = coef.shape[0]
    vh = np.zeros((B, n_fine // 2 + 1), complex)
    # Rescale rfft coefficients from the coarse grid's length to the fine one.
    vh[:, : coef.shape[-1]] = coef * (n_fine / N_GRID)
    k = np.arange(n_fine // 2 + 1) * (2 * np.pi / L_DOMAIN)
    keep = (np.arange(n_fine // 2 + 1) < n_fine // 3).astype(float)
    h = dt / sub
    E = np.exp(-NU * k**2 * h / 2)
    E2 = E * E

    def N(v):
        u = np.fft.irfft(v, n=n_fine, axis=-1)
        return -0.5j * k * np.fft.rfft(u * u, axis=-1) * keep

    stride = n_fine // N_GRID
    out = [np.fft.irfft(vh, n=n_fine, axis=-1)[:, ::stride]]
    for _ in range(n_steps):
        for _ in range(sub):
            a = h * N(vh)
            b = h * N(E * (vh + a / 2))
            c = h * N(E * vh + b / 2)
            d = h * N(E2 * vh + E * c)
            vh = E2 * vh + (E2 * a + 2 * E * (b + c) + d) / 6
        out.append(np.fft.irfft(vh, n=n_fine, axis=-1)[:, ::stride])
    res = np.stack(out, axis=1)
    require(bool(np.all(np.isfinite(res))), "Burgers reference not finite")
    return res


def solve(name: str, coef: np.ndarray, n_steps: int) -> np.ndarray:
    sys = PDES[name]
    if name == "burgers":
        return burgers_reference(coef, n_steps, sys.dt)
    return exact_linear(name, coef, n_steps, sys.dt)


def burgers_convergence(n_ics: int = 3, seed: int = 0) -> dict:
    """Refine the Burgers reference in space and time, compare at the coarse
    points at the end of a test rollout, in units of the initial RMS."""
    sys = PDES["burgers"]
    coef = sample_ic(np.random.default_rng(seed), n_ics, "ood", "burgers")
    n = sys.test_steps
    best = burgers_reference(coef, n, sys.dt, n_fine=2 * N_FINE, sub=50)
    rms0 = best[:, 0].std(axis=-1)
    rows = []
    for nf, sub in ((256, 25), (512, 25), (N_FINE, 12), (N_FINE, 25), (2 * N_FINE, 25)):
        tr = burgers_reference(coef, n, sys.dt, n_fine=nf, sub=sub)
        err = float(np.max(np.abs(tr - best) / rms0[:, None, None]))
        rows.append({"n_fine": nf, "substeps": sub, "max_err": err})
    used = next(r for r in rows if r["n_fine"] == N_FINE and r["substeps"] == 25)
    return {
        "rows": rows,
        "used": {"n_fine": N_FINE, "substeps": 25},
        "used_max_err": used["max_err"],
        "against": {"n_fine": 2 * N_FINE, "substeps": 50},
    }


def physics_step(name: str) -> Callable:
    """The coarse-grid textbook scheme as a torch map u_n -> u_{n+1}."""
    sys = PDES[name]
    h = sys.dt / sys.physics_substeps

    def rhs(u):
        return sys.physics_rhs(u, _TorchLib)

    def step(u):
        for _ in range(sys.physics_substeps):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * h * k1)
            k3 = rhs(u + 0.5 * h * k2)
            k4 = rhs(u + h * k3)
            u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    return step


# ---------------------------------------------------------------------------


@dataclass
class PdeData:
    system: str
    train: np.ndarray  # (n_traj, train_steps+1, N)
    val: np.ndarray
    test: np.ndarray
    ood: np.ndarray
    mean: float
    std: float
    dstd: float
    rscale: float  # spread of what the known term misses (closure systems)


def pde_fixed_sets(
    name: str, n_test: int = 16, n_val: int = 8, n_steps: int | None = None
) -> dict[str, np.ndarray]:
    n = n_steps or PDES[name].test_steps
    val = solve(name, sample_ic(np.random.default_rng(999), n_val, "val", name), n)
    trng = np.random.default_rng(1000)
    test = solve(name, sample_ic(trng, n_test, "test", name), n)
    ood = solve(name, sample_ic(trng, n_test, "ood", name), n)
    return {"val": val, "test": test, "ood": ood}


def pde_train_set(name: str, n_traj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return solve(name, sample_ic(rng, n_traj, "train", name), PDES[name].train_steps)


def pde_assemble(name: str, train: np.ndarray, fixed: dict) -> PdeData:
    sys = PDES[name]
    inc = (train[:, 1:] - train[:, :-1]) / sys.dt
    rscale = float(inc.std())
    if sys.known_rhs is not None:
        import torch

        kn = sys.known_rhs(torch.as_tensor(train[:, :-1])).numpy()
        rscale = float((inc - kn).std())
    return PdeData(
        name,
        train,
        fixed["val"],
        fixed["test"],
        fixed["ood"],
        float(train.mean()),
        float(train.std()),
        float(inc.std()),
        rscale,
    )


def make_pde_data(
    name: str, n_traj: int, seed: int, n_test: int = 16, n_val: int = 8
) -> PdeData:
    return pde_assemble(
        name, pde_train_set(name, n_traj, seed), pde_fixed_sets(name, n_test, n_val)
    )
