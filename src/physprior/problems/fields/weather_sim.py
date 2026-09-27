"""Simulated control for fields/weather: a temperature field whose law and
residual are known, at the real station positions.

    T = T0 + a dlon + b dlat + Gamma z  +  r(x)  +  e  [ - I(z) ]

r      a draw from a zero-mean GP with the kernel of `kriging.py`
       (s_f = 1 K, l_h = 40 km, l_z = 0.4 km): weather structure the law
       does not describe, spatially correlated.
e      independent noise, 0.3 K: station siting and sampling error.
I(z)   optional cold pool, C max(0, 1 - z / z_top), C = 6 K, z_top =
       1.2 km. Below z_top temperature rises with height by C / z_top
       = 5 K/km relative to the law: a valley inversion.

The constants were fixed a priori, not fitted to the real field. They are
of the size reported for the Alps (lapse rates -4 to -7 K/km, winter
cold-pool deficits of several K below 1-1.5 km; Rolland 2003; Kirchner et
al. 2013), which is all they need to be.

Two cases:
  law        the law is complete up to r and e. Any error in the recovered
             Gamma is the method's.
  inversion  the law is incomplete in the way the January data is expected
             to be. The best linear Gamma is then not the free-air Gamma,
             and the gap is known.

The coverage study asks a question the real data cannot: is the physics
arm's curve_fit error bar, which assumes independent residuals, calibrated when
the residual is spatially correlated? Universal kriging models the
correlation, so its GLS error bar should be.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from physprior.benchmark.protocol import Problem, fit_arm, score
from physprior.io import save_json, save_table
from physprior.methods.base import REPORT_SEEDS
from physprior.units import require

from . import kriging
from . import weather as W

TRACK_ROOT = "fields/weather/sim"
TRUTH = {"T0": 16.0, "a": -0.3, "b": -1.0, "Gamma": -5.0}  # K, K/deg, K/deg, K/km
RESIDUAL = {"s_f": 1.0, "l_h_km": 40.0, "l_z_km": 0.4}
NOISE_K = 0.3
INVERSION = {"C": 6.0, "z_top_km": 1.2}
CASES = ("law", "inversion")
COVERAGE_DRAWS = 30
ARMS = ("oracle", "physics", "pinn", "pinn_static", "sr", "nn")


def track(case: str) -> str:
    return f"{TRACK_ROOT}/{case}"


def cold_pool(z_km: np.ndarray, C: float, z_top_km: float) -> np.ndarray:
    return C * np.clip(1.0 - np.asarray(z_km) / z_top_km, 0.0, None)


def best_linear_gamma(x: np.ndarray, case: str) -> float:
    """The Gamma of the least-squares law fit to the NOISE-FREE mean field.

    With the cold pool on, this is what a perfect linear fit would return at
    these stations. It differs from TRUTH["Gamma"] by the pool, not by any
    method.
    """
    mean = W.law_np(x, **TRUTH)
    if case == "inversion":
        mean = mean - cold_pool(x[:, 0], **INVERSION)
    coef, *_ = np.linalg.lstsq(W.law_basis(x), mean, rcond=None)
    return float(coef[3])


def simulate(x: np.ndarray, case: str, seed: int) -> dict[str, np.ndarray]:
    """One realisation at stations `x` = (z_km, lon, lat)."""
    require(case in CASES, f"unknown sim case {case!r}")
    require(bool(np.all(np.abs(x[:, 0]) < 9.0)), "sim: height must be in km")
    rng = np.random.default_rng(seed)
    X = kriging.to_km(x, W.LON0, W.LAT0)
    K = kriging.kernel(X, X, {**RESIDUAL, "s_n": 0.0}) + 1e-9 * np.eye(len(x))
    r = np.linalg.cholesky(K) @ rng.standard_normal(len(x))
    e = rng.normal(0.0, NOISE_K, len(x))
    mean = W.law_np(x, **TRUTH)
    if case == "inversion":
        mean = mean - cold_pool(x[:, 0], **INVERSION)
    return {"y": mean + r + e, "mean": mean, "residual": r, "noise": e}


def problem(x: np.ndarray, case: str, seed: int) -> tuple[Problem, dict]:
    sim = simulate(x, case, seed)
    prob = W.make_problem(
        track(case),
        x,
        sim["y"],
        {"Gamma": TRUTH["Gamma"]},
        W.NN_CFG["july"],
        notes=f"simulated field, case {case}, seed {seed}",
    )
    return prob, sim


def station_positions() -> np.ndarray:
    """The July station set: real positions, real heights."""
    fld = W.load_field("july")
    return np.column_stack([fld.elev_m / 1000.0, fld.lon, fld.lat])


def study_case(x: np.ndarray, case: str, seeds=REPORT_SEEDS, arms=ARMS) -> pd.DataFrame:
    """Elevation split, every arm and both krigings, one realisation per seed."""
    rows = []
    g_lin = best_linear_gamma(x, case)
    for seed in seeds:
        prob, _ = problem(x, case, seed)
        itr, ite = W.elevation_split(prob)
        fits = [fit_arm(a, prob, itr, seed, sr_fast=True) for a in arms]
        fits += [W.fit_kriging(prob, itr, k) for k in ("gp", "uk")]
        for f in fits:
            row = score(prob, f, itr, ite, sweep="elevation", seed=seed, case=case)
            row["dTdz"] = W.effective_lapse_rate(f, prob.x)
            row["gamma_true"] = TRUTH["Gamma"]
            row["gamma_best_linear"] = g_lin
            rows.append(row)
    return pd.DataFrame(rows)


def coverage(x: np.ndarray, case: str, draws: int = COVERAGE_DRAWS) -> pd.DataFrame:
    """Gamma and its quoted sigma from physics and uk, over many draws, all
    stations. Draw seeds 1000.. are simulation seeds, not tuning seeds."""
    g_ref = best_linear_gamma(x, case)
    rows = []
    for k in range(draws):
        seed = 1000 + k
        prob, _ = problem(x, case, seed)
        idx = np.arange(len(prob))
        for f in (fit_arm("physics", prob, idx, seed), W.fit_kriging(prob, idx, "uk")):
            g, s = f.params["Gamma"], f.param_sigma["Gamma"]
            rows.append(
                {
                    "case": case,
                    "draw": seed,
                    "arm": f.name,
                    "gamma": g,
                    "sigma": s,
                    "gamma_ref": g_ref,
                    "z": (g - g_ref) / s,
                }
            )
    return pd.DataFrame(rows)


def coverage_summary(cov: pd.DataFrame) -> pd.DataFrame:
    """Fraction of draws with |Gamma - ref| < 1 sigma (0.683 if calibrated), and
    the spread of the pulls (1 if calibrated)."""
    g = cov.groupby(["case", "arm"])
    return pd.DataFrame(
        {
            "draws": g.size(),
            "gamma_ref": g.gamma_ref.first(),
            "gamma_mean": g.gamma.mean(),
            "gamma_sd_over_draws": g.gamma.std(),
            "sigma_quoted_median": g.sigma.median(),
            "coverage_1sigma": g.z.apply(lambda z: float(np.mean(np.abs(z) < 1.0))),
            "pull_sd": g.z.std(),
        }
    ).reset_index()


def run(quick: bool = False) -> dict:
    x = station_positions()
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    arms = ARMS if not quick else ("oracle", "physics", "pinn", "nn")
    draws = COVERAGE_DRAWS if not quick else 5
    meta: dict = {
        "truth": TRUTH,
        "residual": RESIDUAL,
        "noise_k": NOISE_K,
        "inversion": INVERSION,
        "n_stations": len(x),
        "gamma_best_linear": {c: best_linear_gamma(x, c) for c in CASES},
    }
    covs = []
    for case in CASES:
        t0 = time.time()
        df = study_case(x, case, seeds=seeds, arms=arms)
        save_table(df, track(case), "extrapolation")
        covs.append(coverage(x, case, draws))
        print(f"[{track(case)}] done ({time.time() - t0:.0f} s)", flush=True)
    cov = pd.concat(covs, ignore_index=True)
    save_table(cov, TRACK_ROOT, "coverage")
    summ = coverage_summary(cov)
    save_table(summ, TRACK_ROOT, "coverage_summary")
    meta["coverage"] = summ.to_dict(orient="records")
    save_json(meta, TRACK_ROOT, "meta")
    return meta
