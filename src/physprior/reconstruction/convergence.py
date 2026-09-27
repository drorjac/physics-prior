"""Convergence studies for every solver the reconstruction study uses.

1. `series_truncation`  the box sine series against its converged value, as
                        a function of the modes kept per axis.
2. `fd_vs_series`       the second-order FD solver against the series, as a
                        function of the grid spacing (two independent
                        methods for the same field; the observed order
                        should approach 2).
3. `fd_manufactured`    the FD solver against manufactured solutions,
                        Dirichlet in 1/2/3-D and the insulated-wall (Neumann)
                        plate that the model-mismatch study uses as truth.
4. `free_residual`      the closed-form free-space potentials inserted in a
                        central-difference Laplacian: -lap(Phi) - g -> 0 as h^2.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import fields as F

SERIES_KM = (6, 8, 10, 12, 14, 16, 18, 20, 24, 28)
FD_N = {1: (16, 32, 64, 128, 256), 2: (8, 16, 32, 64, 128), 3: (8, 12, 16, 24, 32, 48)}
FD_N_QUICK = {1: (16, 32, 64), 2: (8, 16, 32), 3: (4, 6, 8)}
FREE_H = (4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3)


def _order(df: pd.DataFrame, hcol: str, ecol: str) -> pd.DataFrame:
    df = df.sort_values(hcol, ascending=False).copy()
    e, h = df[ecol].to_numpy(), df[hcol].to_numpy()
    order = np.full(len(df), np.nan)
    order[1:] = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    df["observed_order"] = order
    return df


def series_truncation(seed: int = 3) -> pd.DataFrame:
    rows = []
    for d in (1, 2, 3):
        f = F.draw_field("box", d, seed)
        x = np.random.default_rng([seed, d]).uniform(0, 1, (400, d))
        ref = f.value(x)
        for km in SERIES_KM:
            g = replace(f.geom, km=km)
            v = g.lift(x) @ f.lift_coef
            for c, q in zip(f.centers, f.q, strict=True):
                v = v + q * g.green(x, c)[0]
            rows.append(
                {
                    "d": d,
                    "kmax": km,
                    "rel_max_err": float(np.max(np.abs(v - ref)) / np.std(ref)),
                    "is_fit_kmax": km == F.KMAX_FIT[d],
                }
            )
    return pd.DataFrame(rows)


def fd_vs_series(seed: int = 3, quick: bool = False) -> pd.DataFrame:
    rows = []
    grids = FD_N_QUICK if quick else FD_N
    for d in (1, 2, 3):
        f = F.draw_field("box", d, seed)
        part = []
        for n in grids[d]:
            g1, u = F.solve_fd(
                d, n, f.source, lambda z, f=f: f.geom.lift(z) @ f.lift_coef
            )
            X = np.stack(np.meshgrid(*([g1] * d), indexing="ij"), -1).reshape(-1, d)
            us = f.value(X)
            part.append(
                {
                    "d": d,
                    "n": n,
                    "h": 1.0 / n,
                    "rel_max_err": float(np.max(np.abs(u.ravel() - us)) / np.std(us)),
                }
            )
        rows.append(_order(pd.DataFrame(part), "h", "rel_max_err"))
    return pd.concat(rows, ignore_index=True)


_MMS_LIFT = np.array([0.3, -0.2, 0.5, 0.1])


def _mms(d: int, neumann: bool):
    """(exact, source, boundary) of a manufactured solution."""
    if neumann:
        # sin(pi x / 2) has zero slope at x = 1: the insulated wall
        def exact(X):
            return np.sin(0.5 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

        def src(X):
            return 1.25 * np.pi**2 * exact(X)

        return exact, src, exact

    def exact_d(X):
        bump = np.prod(np.sin(np.pi * X), axis=1)
        return bump + _MMS_LIFT[0] + X @ _MMS_LIFT[1 : d + 1]

    def src_d(X):
        return d * np.pi**2 * np.prod(np.sin(np.pi * X), axis=1)

    return exact_d, src_d, exact_d


def fd_manufactured(quick: bool = False) -> pd.DataFrame:
    rows = []
    grids = FD_N_QUICK if quick else FD_N
    for d in (1, 2, 3):
        cases = [("dirichlet", False)] + ([("neumann_x1", True)] if d == 2 else [])
        for name, neu in cases:
            exact, src, bnd = _mms(d, neu)
            part = []
            for n in grids[d]:
                g1, u = F.solve_fd(d, n, src, bnd, neumann_x1=neu)
                X = np.stack(np.meshgrid(*([g1] * d), indexing="ij"), -1).reshape(-1, d)
                part.append(
                    {
                        "d": d,
                        "boundary": name,
                        "n": n,
                        "h": 1.0 / n,
                        "max_err": float(np.max(np.abs(u.ravel() - exact(X)))),
                    }
                )
            rows.append(_order(pd.DataFrame(part), "h", "max_err"))
    return pd.concat(rows, ignore_index=True)


def free_residual() -> pd.DataFrame:
    rows = []
    w = F.SOURCE_WIDTH
    for d in (1, 2, 3):
        x = np.random.default_rng(d).uniform(-3 * w, 3 * w, (300, d))
        c = np.zeros(d)
        g0 = F.gaussian(c[None], c, w)[0]
        part = []
        for h in FREE_H:
            lap = np.zeros(len(x))
            for i in range(d):
                e = h * np.eye(d)[i]
                lap += (
                    F.free_green(x + e, c, w=w)[0]
                    - 2 * F.free_green(x, c, w=w)[0]
                    + F.free_green(x - e, c, w=w)[0]
                ) / h**2
            err = np.max(np.abs(-lap - F.gaussian(x, c, w))) / g0
            part.append({"d": d, "h": h, "rel_max_residual": float(err)})
        rows.append(_order(pd.DataFrame(part), "h", "rel_max_residual"))
    return pd.concat(rows, ignore_index=True)


def run_all(quick: bool = False) -> dict[str, pd.DataFrame]:
    return {
        "convergence_series": series_truncation(),
        "convergence_fd_series": fd_vs_series(quick=quick),
        "convergence_fd_mms": fd_manufactured(quick=quick),
        "convergence_free": free_residual(),
    }
