"""ODE integrators for the mechanics simulations.

Two families, on purpose:

`velocity_verlet`  symplectic. It does not conserve energy exactly, but its
                   energy error OSCILLATES and stays bounded forever, because
                   it exactly conserves a nearby "shadow" Hamiltonian. This is
                   what orbit integrations use.

`rk4`              higher local accuracy, not symplectic. Its energy error
                   DRIFTS, secularly, and a bound orbit slowly spirals.

Both are here because the comparison is the point. It is the same lesson the
Mercury track paid for with a false 56-sigma refutation of general relativity:
the numerical method is part of the physics model, and a result you have not
checked for convergence is not a result.

Everything is plain numpy so the trajectories can be handed straight to the
symbolic-regression and PINN arms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# accel(r) -> (N, d) acceleration given (N, d) positions
AccelFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Trajectory:
    """The output of a run, in SI unless the caller says otherwise."""

    t: np.ndarray  # (T,)
    r: np.ndarray  # (T, N, d) positions
    v: np.ndarray  # (T, N, d) velocities
    masses: np.ndarray  # (N,)
    method: str
    dt: float
    names: list[str] | None = None
    # Always populated by `integrate`; the default keeps the dataclass usable
    # when a Trajectory is built by hand.
    meta: dict = field(default_factory=dict)

    @property
    def n_bodies(self) -> int:
        return self.r.shape[1]

    def energy(self, G: float) -> np.ndarray:
        """Total energy at each step. Constant is the whole point."""
        ke = 0.5 * np.sum(self.masses[None, :, None] * self.v**2, axis=(1, 2))
        return ke + potential_energy(self.r, self.masses, G)

    def angular_momentum(self) -> np.ndarray:
        """|L| at each step, about the origin (3-D; 2-D is embedded)."""
        r, v = _to3(self.r), _to3(self.v)
        L = np.sum(self.masses[None, :, None] * np.cross(r, v), axis=1)
        return np.linalg.norm(L, axis=1)

    def drift(self, G: float) -> dict:
        """Relative drift of the conserved quantities over the whole run."""
        e, L = self.energy(G), self.angular_momentum()
        return {
            "energy_rel_drift": float(abs(e[-1] - e[0]) / abs(e[0])),
            "energy_rel_spread": float(np.ptp(e) / abs(e[0])),
            "L_rel_drift": float(abs(L[-1] - L[0]) / abs(L[0])) if L[0] else 0.0,
        }


def _to3(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 3:
        return x
    pad = np.zeros((*x.shape[:-1], 3 - x.shape[-1]))
    return np.concatenate([x, pad], axis=-1)


def potential_energy(r: np.ndarray, masses: np.ndarray, G: float) -> np.ndarray:
    """Newtonian potential energy, summed over unordered pairs."""
    single = r.ndim == 2
    R = r[None] if single else r
    T, N, _ = R.shape
    out = np.zeros(T)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(R[:, i] - R[:, j], axis=-1)
            out -= G * masses[i] * masses[j] / d
    return out[0] if single else out


def gravity_accel(masses: np.ndarray, G: float, softening: float = 0.0) -> AccelFn:
    """Newtonian pairwise acceleration, a_i = -G sum_j m_j (r_i-r_j)/|r_i-r_j|^3.

    `softening` is 0 by default. It is there for the chaotic three-body runs,
    where two bodies can pass arbitrarily close and the true acceleration
    diverges; a nonzero value changes the physics and is always reported.
    """

    def accel(r: np.ndarray) -> np.ndarray:
        d = r[None, :, :] - r[:, None, :]  # d[i,j] = r_j - r_i
        dist2 = np.sum(d * d, axis=-1) + softening**2
        np.fill_diagonal(dist2, np.inf)  # no self-interaction
        inv = dist2**-1.5
        return G * np.einsum(
            "ij,ij,ijk->ik", masses[None, :] * np.ones_like(inv), inv, d
        )

    return accel


def velocity_verlet(
    accel: AccelFn,
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    stride: int = 1,
) -> tuple[np.ndarray, ...]:
    r, v = np.array(r0, float), np.array(v0, float)
    a = accel(r)
    keep = range(0, n_steps + 1, stride)
    R = np.empty((len(keep), *r.shape))
    V = np.empty_like(R)
    T = np.empty(len(keep))
    k = 0
    for step in range(n_steps + 1):
        if step % stride == 0:
            R[k], V[k], T[k] = r, v, step * dt
            k += 1
        v_half = v + 0.5 * dt * a
        r = r + dt * v_half
        a = accel(r)
        v = v_half + 0.5 * dt * a
    return T, R, V


def rk4(
    accel: AccelFn,
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    stride: int = 1,
) -> tuple[np.ndarray, ...]:
    r, v = np.array(r0, float), np.array(v0, float)
    keep = range(0, n_steps + 1, stride)
    R = np.empty((len(keep), *r.shape))
    V = np.empty_like(R)
    T = np.empty(len(keep))
    k = 0
    for step in range(n_steps + 1):
        if step % stride == 0:
            R[k], V[k], T[k] = r, v, step * dt
            k += 1
        k1r, k1v = v, accel(r)
        k2r, k2v = v + 0.5 * dt * k1v, accel(r + 0.5 * dt * k1r)
        k3r, k3v = v + 0.5 * dt * k2v, accel(r + 0.5 * dt * k2r)
        k4r, k4v = v + dt * k3v, accel(r + dt * k3r)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return T, R, V


METHODS = {"verlet": velocity_verlet, "rk4": rk4}


def integrate(
    masses,
    r0,
    v0,
    dt,
    n_steps,
    G,
    method="verlet",
    stride=1,
    softening=0.0,
    names=None,
    meta=None,
) -> Trajectory:
    masses = np.asarray(masses, float)
    accel = gravity_accel(masses, G, softening)
    t, R, V = METHODS[method](
        accel, np.asarray(r0, float), np.asarray(v0, float), dt, n_steps, stride
    )
    return Trajectory(
        t=t, r=R, v=V, masses=masses, method=method, dt=dt, names=names, meta=meta or {}
    )
