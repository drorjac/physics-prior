"""Scoring a stepper by iterating it on its own output.

Errors are in standardised units: for ODEs each component is divided by its
training-set standard deviation, for PDEs the field is divided by the RMS of
the initial fluctuation of that trajectory. A rollout that leaves the region
the data lives in by a wide margin (|z| > BLOWUP_Z) or produces a non-finite
number is marked as blown up and its later states are infinite.

Thresholds were fixed before any model was trained: a rollout is "valid"
while its error is below VALID_THR (0.1 of a standard deviation), and for the
chaotic Lorenz system while it is below LORENZ_THR = 0.4, the threshold of
Pathak et al., Phys. Rev. Lett. 120, 024102 (2018).
"""

from __future__ import annotations

import numpy as np
import torch

from .steppers import DTYPE

VALID_THR = 0.1
LORENZ_THR = 0.4
BLOWUP_Z = 50.0


def rollout(model, u0: np.ndarray, n_steps: int, mean, std) -> np.ndarray:
    """Iterate `model` from u0 (B, ...). Returns (B, n+1, ...); inf after a
    blow-up."""
    mu = np.asarray(mean, float)
    sd = np.asarray(std, float)
    u = torch.as_tensor(np.asarray(u0, float), dtype=DTYPE)
    out = np.full((u.shape[0], n_steps + 1, *u.shape[1:]), np.inf)
    out[:, 0] = u0
    alive = np.ones(u.shape[0], bool)
    need = getattr(model, "needs_grad", False)
    model.eval()
    for k in range(1, n_steps + 1):
        if need:
            with torch.enable_grad():
                u = model(u.detach().requires_grad_(True)).detach()
        else:
            with torch.no_grad():
                u = model(u)
        x = u.numpy()
        z = np.abs((x - mu) / sd).reshape(len(x), -1)
        bad = ~np.isfinite(z).all(axis=1) | (
            np.nan_to_num(z, nan=np.inf).max(axis=1) > BLOWUP_Z
        )
        alive &= ~bad
        if not alive.any():
            break
        out[alive, k] = x[alive]
        # Freeze blown rollouts at a harmless state so the batch keeps going.
        if bad.any():
            x = x.copy()
            x[bad] = np.asarray(u0)[bad]
            u = torch.as_tensor(x, dtype=DTYPE)
    return out


def error_curve(pred: np.ndarray, truth: np.ndarray, scale) -> np.ndarray:
    """(B, T) RMS error per step, in units of `scale` (broadcast over state)."""
    e = (pred - truth) / scale
    with np.errstate(invalid="ignore", over="ignore"):
        c = np.sqrt(np.mean(e.reshape(*e.shape[:2], -1) ** 2, axis=-1))
    return np.where(np.isfinite(c), c, np.inf)


def valid_steps(curve: np.ndarray, thr: float) -> np.ndarray:
    """Per rollout: the number of steps before the error first exceeds thr."""
    over = curve > thr
    first = np.where(over.any(axis=1), over.argmax(axis=1), curve.shape[1])
    return first - 1


def onestep_error(model, traj: np.ndarray, scale, max_steps: int = 200) -> float:
    """Median over trajectories of the RMS one-step error on true states."""
    n = min(max_steps, traj.shape[1] - 1)
    x = traj[:, :n]
    y = traj[:, 1 : n + 1]
    flat = torch.as_tensor(x.reshape(-1, *x.shape[2:]), dtype=DTYPE)
    need = getattr(model, "needs_grad", False)
    model.eval()
    if need:
        with torch.enable_grad():
            p = model(flat.requires_grad_(True)).detach().numpy()
    else:
        with torch.no_grad():
            p = model(flat).numpy()
    p = p.reshape(y.shape)
    e = error_curve(p, y, scale)
    return float(np.median(np.sqrt(np.mean(e**2, axis=1))))


def invariant_drift(pred: np.ndarray, fn, kind: str) -> np.ndarray:
    """|I(t) - I(0)| / normaliser for each rollout: (B, T)."""
    with np.errstate(invalid="ignore", over="ignore"):
        inv = fn(pred)
        i0 = inv[:, :1]
        if kind == "H":
            # pendulum: energy above the bottom of the well, H_min = -1
            norm = i0 + 1.0
        else:
            norm = np.abs(i0)
        d = np.abs(inv - i0) / norm
    return np.where(np.isfinite(d), d, np.inf)


def window_mean(x: np.ndarray, frac: float, last: bool) -> np.ndarray:
    n = max(1, round(x.shape[1] * frac))
    seg = x[:, -n:] if last else x[:, 1 : n + 1]
    with np.errstate(invalid="ignore"):
        return np.mean(seg, axis=1)


def log_indices(n: int, k: int = 60) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, n, k)).astype(int))
