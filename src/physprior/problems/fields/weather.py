"""Track fields/weather -- near-surface air temperature over the Alps.

DATA     NOAA ISD-Lite hourly station records (data/sources/isd.py), in the
         box lon 5.8..13.5 E, lat 45.5..48.0 N: Switzerland, western Austria,
         the Italian Alps and their forelands, from sea level to 3576 m.

SNAPSHOT The rule was fixed before any fit was run, and is the same for both
         cases: the mean over one calendar month of 2023 of the 12 UTC air
         temperature, for every station with a 12 UTC reading on at least
         80% of the days of that month; co-located duplicate records merged.
         Two months:
           july      2023-07. A well-mixed summer boundary layer.
           january   2023-01. The harder case: winter cold pools and valley
                     inversions, in which temperature rises with height over
                     the lowest few hundred metres and the law's single
                     linear lapse rate is known to be wrong.

LAW      T = T0 + a (lon - lon0) + b (lat - lat0) + Gamma z,   z in km.

         x = (z_km, lon, lat). Height comes first so that the protocol's
         standard extrapolation split (low end of column 0 against the high
         end) is the elevation extrapolation this track is about.

PARAM    Gamma, the lapse rate in K/km. Published reference: the ICAO / ISO
         2533 standard atmosphere, -6.5 K/km in the troposphere (ICAO Doc
         7488/3, 1993; ISO 2533:1975). That is a free-atmosphere value. The
         near-surface lapse rate measured along a mountain slope differs from
         it -- it is typically shallower in winter and close to or steeper
         than -6.5 in summer afternoons (Rolland 2003, J. Climate 16, 1032;
         Kirchner et al. 2013, Hydrol. Earth Syst. Sci. 17, 4291) -- so the
         comparison is to a reference, not to a truth. T0, a and b have no
         published value.

ORACLE   The protocol's oracle is the law with published constants. Only
         Gamma has one, so here the oracle holds Gamma at -6.5 K/km and fits
         T0, a and b on the training split. It is the standard-atmosphere
         lapse rate put to the test, not a ceiling.

PINN     The project's PINN, with its constants scaled so the optimiser can
         reach them: T0 starts at the training mean, Gamma is optimised in
         log space from -1 K/km (see `pinn_params`). The same PINN from the
         fixed starts of `params()` is reported as `pinn_static`: its
         constants do not reach the least-squares values in 4000 epochs.

SPLITS   elevation  train on the lowest 75% of stations by height, test on
                    the highest 25% (fixed before any fit).
         block      four longitude-quartile blocks; each is held out in turn
                    and predicted from the other three.

EXTRA    ordinary kriging (GP, constant mean) and universal kriging (the law
         as the GP mean) on the same splits, reported beside the arms as the
         helium track reports the Rydberg-Ritz fit. Deterministic: one fit.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from physprior.benchmark.protocol import (
    Problem,
    fit_arm,
    law,
    score,
    split_extrapolate,
    split_random,
    sweep_budget,
    sweep_noise,
)
from physprior.data.sources import isd
from physprior.io import save_json, save_table
from physprior.methods import pinn as pinn_mod
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS, Fit
from physprior.methods.pinn import PhysParam
from physprior.units import require

from . import kriging

TRACK_ROOT = "fields/weather"
CASES = {
    "july": isd.SnapshotRule(year=2023, month=7),
    "january": isd.SnapshotRule(year=2023, month=1),
}
# Box centre, so T0 is the sea-level temperature at the middle of the box.
LON0, LAT0 = 9.65, 46.75
# ICAO standard atmosphere, troposphere: -6.5 K/km (ICAO Doc 7488/3, 1993;
# ISO 2533:1975).
GAMMA_STD = -6.5
ELEV_TRAIN_FRAC = 0.75
N_BLOCKS = 4
ARMS = ("oracle", "physics", "pinn", "sr", "nn")

# The `pinn` arm's physics weight per case, chosen on the tuning seeds by
# `physprior.problems.fields.retune` (results/fields/weather/<case>/tune/).
# July is pinned at the top of the extended grid: its validation block
# prefers no correction, so the arm is the law by choice.
PINN_W_PHYS = {"july": 1e6, "january": 0.01}
# the elevation split also runs the fixed-start PINN, as a diagnostic
ELEV_ARMS = (*ARMS, "pinn_static")
SWEEP_CASE = "july"
# Set for compute before any run, not tuned: the law here is linear and the
# PINN's loss has flattened well before this on the tuning seeds.
PINN_EPOCHS = 4000

# Chosen by `tune_nn(case, epochs=2000)` on the tuning seeds (3/7/19), on a
# random half of the elevation-split training stations, before any reporting
# seed was run. Validation RMSE: july 0.953 degC, january 0.923 degC.
NN_CFG: dict[str, dict[str, Any]] = {
    "july": dict(width=16, depth=2, weight_decay=1e-4, epochs=2000),
    "january": dict(width=16, depth=2, weight_decay=1e-2, epochs=2000),
}


def track(case: str) -> str:
    return f"{TRACK_ROOT}/{case}"


# ---------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------


@law("T0 + a*(lon - lon0) + b*(lat - lat0) + Gamma*z")
def law_np(x, T0, a, b, Gamma):
    x = np.atleast_2d(np.asarray(x, float))
    return T0 + a * (x[:, 1] - LON0) + b * (x[:, 2] - LAT0) + Gamma * x[:, 0]


def law_t(x, T0, a, b, Gamma):
    return T0 + a * (x[:, 1] - LON0) + b * (x[:, 2] - LAT0) + Gamma * x[:, 0]


def law_basis(x: np.ndarray) -> np.ndarray:
    """The law is linear in its constants: [1, dlon, dlat, z]."""
    x = np.asarray(x, float)
    return np.column_stack([np.ones(len(x)), x[:, 1] - LON0, x[:, 2] - LAT0, x[:, 0]])


def const_basis(x: np.ndarray) -> np.ndarray:
    return np.ones((len(x), 1))


PARAM_NAMES = ("T0", "a", "b", "Gamma")


def params() -> list[PhysParam]:
    """The constants to recover, with fixed starting values.

    Gamma starts at zero, not at the published value, so a recovered -6.5
    cannot be an initialisation that never moved. These are what `physics`
    starts from (it is linear least squares, so the start does not matter)
    and what the `pinn_static` diagnostic uses.
    """
    return [
        PhysParam("T0", 10.0, positive=False, lo=-50.0, hi=50.0),
        PhysParam("a", 0.0, positive=False, lo=-10.0, hi=10.0),
        PhysParam("b", 0.0, positive=False, lo=-10.0, hi=10.0),
        PhysParam("Gamma", 0.0, positive=False, lo=-30.0, hi=30.0),
    ]


def pinn_params(y_train: np.ndarray) -> list[PhysParam]:
    """The PINN's constants, scaled so Adam can reach them.

    Adam moves an additive parameter by about one learning rate per step
    whatever its size. With lr 5e-3 and a cosine schedule over 4000 epochs
    a constant can travel about 10 units in total, which is less than the
    distance from a fixed T0 = 10 degC to a July sea-level temperature. So:

    T0      starts at the mean of the training temperatures -- a statistic
            of the training data, like the standardisation every network
            arm applies to its targets;
    Gamma   is multiplicative, Gamma = -1 K/km * exp(r): log-space, the
            project's convention for constants of fixed sign
            (`PhysParam.positive` means multiplicative; the sign is the
            initial value's). This assumes temperature falls with height
            on average over the box. -1 K/km is an order of magnitude, not
            the published value.

    Chosen on the tuning seeds (3/7/19) by one criterion: that the PINN's
    constants, trained on the elevation training split, reach the
    least-squares constants on the same split (they agree to 1e-3 K/km;
    with `params()` they stopped at -0.05 K/km in July). No held-out number
    was used. The fixed-start version is reported as `pinn_static`.
    """
    return [
        PhysParam("T0", float(np.mean(y_train)), positive=False, lo=-50.0, hi=50.0),
        PhysParam("a", 0.0, positive=False, lo=-10.0, hi=10.0),
        PhysParam("b", 0.0, positive=False, lo=-10.0, hi=10.0),
        PhysParam("Gamma", -1.0, positive=True, lo=-30.0, hi=30.0),
    ]


def _pinn(prob, idx, seed, w_phys, plist, name, options):
    xtr, ytr = prob.sub(idx)
    f = pinn_mod.fit_pinn(
        xtr,
        ytr,
        law_t,
        plist,
        w_phys=w_phys,
        seed=seed,
        epochs=prob.pinn_epochs,
        options=options,
        name=name,
    )
    f.expression = f"{getattr(law_np, 'expression', 'law')}  + NN(x)"
    return f


def pinn_impl(
    prob: Problem,
    idx: np.ndarray,
    seed: int,
    w_phys: float,
    options: pinn_mod.PinnOptions = pinn_mod.FROZEN_PINN,
) -> Fit:
    """The project's PINN with the scaled constants of `pinn_params`."""
    return _pinn(prob, idx, seed, w_phys, pinn_params(prob.y[idx]), "pinn", options)


def pinn_static_impl(prob: Problem, idx: np.ndarray, seed: int, w_phys: float) -> Fit:
    """The project's PINN from the fixed starts of `params()`. A diagnostic."""
    return _pinn(prob, idx, seed, w_phys, params(), "pinn_static", pinn_mod.FROZEN_PINN)


def oracle_impl(prob: Problem, idx: np.ndarray, seed: int, w_phys: float) -> Fit:
    """Gamma held at its published value; T0, a, b by least squares."""
    t0 = time.time()
    g = prob.theta_published["Gamma"]
    xtr, ytr = prob.sub(idx)
    H = law_basis(xtr)[:, :3]
    coef, *_ = np.linalg.lstsq(H, ytr - g * xtr[:, 0], rcond=None)
    theta = dict(zip(("T0", "a", "b"), map(float, coef), strict=True)) | {"Gamma": g}
    return Fit(
        name="oracle",
        predict=lambda xq: law_np(xq, **theta),
        params=theta,
        n_free=3,
        seconds=time.time() - t0,
        expression=getattr(law_np, "expression", None),
    )


def _sr_transform(x, y):
    """SR sees (z, lon - lon0, lat - lat0): centred, so its constants are
    O(1-10) and T0 means what it means in the law."""

    def view(xq):
        xq = np.asarray(xq, float)
        return np.column_stack([xq[:, 0], xq[:, 1] - LON0, xq[:, 2] - LAT0])

    def back(xq, raw_predict):
        return raw_predict(view(xq))

    return view(x), np.asarray(y, float), back


def make_problem(
    track_name: str,
    x: np.ndarray,
    y: np.ndarray,
    theta_published: dict[str, float],
    nn_cfg: dict,
    notes: str = "",
) -> Problem:
    """The Problem for any station field with x = (z_km, lon, lat)."""
    return Problem(
        track=track_name,
        x=x,
        y=y,
        law_np=law_np,
        law_t=law_t,
        params=params(),
        theta_published=theta_published,
        xlabel="station height z  [km]",
        ylabel="T  [degC]",
        # Unweighted: the station means have no quoted error, and the law's
        # misfit (degrees) is far above the sampling error of a monthly mean
        # (about 0.3 degC for 30 daily values with a 2 degC day-to-day
        # spread).
        sigma=None,
        sr_transform=_sr_transform,
        sr_kwargs=dict(
            feature_names=["z", "dlon", "dlat"],
            niterations=40,
            maxsize=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "exp"],
        ),
        nn_cfg=dict(nn_cfg),
        pinn_epochs=PINN_EPOCHS,
        arm_impl={
            "oracle": oracle_impl,
            "pinn": pinn_impl,
            "pinn_static": pinn_static_impl,
        },
        notes=notes,
    )


@lru_cache(maxsize=4)
def load_field(case: str) -> isd.StationField:
    fld = isd.monthly_snapshot(CASES[case])
    require(
        bool(np.all((fld.elev_m > -430) & (fld.elev_m < 5000))),
        "weather: station height outside [-430, 5000] m",
    )
    return fld


def problem(case: str) -> tuple[Problem, dict]:
    fld = load_field(case)
    x = np.column_stack([fld.elev_m / 1000.0, fld.lon, fld.lat])
    prob = make_problem(
        track(case),
        x,
        fld.temp_c,
        {"Gamma": GAMMA_STD},
        NN_CFG[case],
        notes=f"NOAA ISD-Lite, {fld.rule.describe()}",
    )
    prob.pinn_w_phys = PINN_W_PHYS[case]
    meta = {
        "case": case,
        "rule": fld.rule.describe(),
        "n_stations": len(fld),
        "z_km_range": [float(x[:, 0].min()), float(x[:, 0].max())],
        "temp_c_range": [float(fld.temp_c.min()), float(fld.temp_c.max())],
        "gamma_published": GAMMA_STD,
        "gamma_reference": "ICAO Doc 7488/3 (1993) / ISO 2533:1975",
        "provenance": fld.provenance,
    }
    return prob, meta


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def elevation_split(prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    return split_extrapolate(prob.x[:, 0], ELEV_TRAIN_FRAC)


def block_splits(prob: Problem) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Longitude quartiles; block k is held out, the other three train."""
    lon = prob.x[:, 1]
    edges = np.quantile(lon, np.linspace(0, 1, N_BLOCKS + 1))
    k = np.clip(np.searchsorted(edges, lon, side="right") - 1, 0, N_BLOCKS - 1)
    idx = np.arange(len(lon))
    return [(b, idx[k != b], idx[k == b]) for b in range(N_BLOCKS)]


# ---------------------------------------------------------------------------
# the lapse rate every arm implies
# ---------------------------------------------------------------------------

LAPSE_STEPS_KM = (0.2, 0.1, 0.05, 0.025)


def effective_lapse_rate(fit: Fit, x: np.ndarray, h: float = 0.05) -> float:
    """Mean dT/dz of the fitted function over the stations `x`, K/km.

    Central difference in z at each station's own position. For the law it
    is Gamma exactly, at any step; for a network it is what the network
    learned about height, which a parameter table cannot show.
    """
    up, dn = x.copy(), x.copy()
    up[:, 0] += h
    dn[:, 0] -= h
    d = (np.asarray(fit.predict(up)) - np.asarray(fit.predict(dn))) / (2 * h)
    return float(np.mean(d))


def lapse_step_study(fit: Fit, x: np.ndarray) -> list[dict]:
    """The derivative's convergence with step: it must stop moving."""
    vals = [effective_lapse_rate(fit, x, h) for h in LAPSE_STEPS_KM]
    return [
        {"arm": fit.name, "h_km": h, "dTdz": v, "change_from_prev": v - p}
        for h, v, p in zip(LAPSE_STEPS_KM, vals, [np.nan, *vals[:-1]], strict=True)
    ]


# ---------------------------------------------------------------------------
# kriging
# ---------------------------------------------------------------------------


def fit_kriging(prob: Problem, idx: np.ndarray, kind: str) -> Fit:
    """kind = "gp" (ordinary kriging) or "uk" (universal: the law as mean)."""
    xtr, ytr = prob.sub(idx)
    if kind == "uk":
        return kriging.fit_gp(
            xtr,
            ytr,
            law_basis,
            lon0=LON0,
            lat0=LAT0,
            name="uk",
            beta_names=PARAM_NAMES,
        )
    require(kind == "gp", f"unknown kriging kind {kind!r}")
    return kriging.fit_gp(xtr, ytr, const_basis, lon0=LON0, lat0=LAT0, name="gp")


def _kriging_row(prob, fit, itr, ite, **tags) -> dict:
    row = score(prob, fit, itr, ite, **tags)
    row["dTdz"] = effective_lapse_rate(fit, prob.x)
    for k, v in fit.extra["hyper"].items():
        row[f"hyper_{k}"] = v
    row["converged"] = fit.extra["converged"]
    row["starts_agreeing"] = fit.extra["starts_agreeing"]
    return row


def kriging_study(prob: Problem, blocks: bool = True) -> pd.DataFrame:
    rows = []
    itr, ite = elevation_split(prob)
    for kind in ("gp", "uk"):
        f = fit_kriging(prob, itr, kind)
        rows.append(_kriging_row(prob, f, itr, ite, sweep="elevation"))
    if blocks:
        for b, itr, ite in block_splits(prob):
            for kind in ("gp", "uk"):
                f = fit_kriging(prob, itr, kind)
                rows.append(_kriging_row(prob, f, itr, ite, sweep="block", block=b))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the arms on the two splits
# ---------------------------------------------------------------------------


def _arm_row(prob, f, itr, ite, **tags) -> dict:
    row = score(prob, f, itr, ite, **tags)
    row["dTdz"] = effective_lapse_rate(f, prob.x)
    return row


def study_elevation(prob: Problem, seeds=REPORT_SEEDS, arms=ARMS) -> pd.DataFrame:
    """The protocol's extrapolation study on the height split, plus the
    lapse rate each arm implies."""
    itr, ite = elevation_split(prob)
    rows = []
    for seed in seeds:
        for arm in arms:
            f = fit_arm(arm, prob, itr, seed)
            rows.append(
                _arm_row(
                    prob,
                    f,
                    itr,
                    ite,
                    sweep="elevation",
                    train_frac=ELEV_TRAIN_FRAC,
                    seed=seed,
                    n_train=len(itr),
                    z_max_train_km=float(prob.x[itr, 0].max()),
                )
            )
    return pd.DataFrame(rows)


def study_blocks(prob: Problem, seeds=REPORT_SEEDS, arms=ARMS) -> pd.DataFrame:
    rows = []
    for b, itr, ite in block_splits(prob):
        for seed in seeds:
            for arm in arms:
                f = fit_arm(arm, prob, itr, seed, sr_fast=True)
                rows.append(
                    _arm_row(
                        prob,
                        f,
                        itr,
                        ite,
                        sweep="block",
                        block=b,
                        seed=seed,
                        n_train=len(itr),
                        lon_lo=float(prob.x[ite, 1].min()),
                        lon_hi=float(prob.x[ite, 1].max()),
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the black box's configuration, chosen on the tuning seeds
# ---------------------------------------------------------------------------


def tune_nn(case: str, epochs: int = 4000) -> dict:
    """Grid-search the MLP on a random half of the elevation training split.

    Only the training stations and the tuning seeds are used, so no
    reported number can have influenced the choice.
    """
    from physprior.methods.neural import tune_mlp

    prob, _ = problem(case)
    itr, _ = elevation_split(prob)
    a_idx, b_idx = split_random(len(itr), len(itr) // 2, seed=TUNE_SEEDS[0])
    a, b = itr[a_idx], itr[b_idx]
    return tune_mlp(
        prob.x[a], prob.y[a], prob.x[b], prob.y[b], TUNE_SEEDS, epochs=epochs
    )


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def headline_fits(prob: Problem, seed: int = REPORT_SEEDS[0]) -> dict:
    """Every station, for the recovered constants."""
    allidx = np.arange(len(prob))
    out: dict[str, Any] = {}
    for arm in ("physics", "pinn"):
        f = fit_arm(arm, prob, allidx, seed)
        out[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "dTdz": effective_lapse_rate(f, prob.x),
        }
    uk = fit_kriging(prob, allidx, "uk")
    out["uk"] = {
        "params": uk.params,
        "sigma": uk.param_sigma,
        "hyper": uk.extra["hyper"],
        "converged": uk.extra["converged"],
    }
    phys = out["physics"]
    out["gamma_minus_published_sigma"] = (phys["params"]["Gamma"] - GAMMA_STD) / phys[
        "sigma"
    ]["Gamma"]
    out["uk_gamma_minus_published_sigma"] = (
        uk.params["Gamma"] - GAMMA_STD
    ) / uk.param_sigma["Gamma"]
    return out


PARTS = ("main", "blocks", "sweeps")


def run_case(case: str, quick: bool = False, parts=PARTS) -> dict:
    """`parts` lets the three independent pieces run in separate processes:
    "main" (elevation split, kriging, derivative check, headline fits and
    meta), "blocks" and "sweeps"."""
    prob, meta = problem(case)
    tr = track(case)
    print(
        f"[{tr}] {len(prob)} stations, z = {prob.x[:, 0].min():.2f}.."
        f"{prob.x[:, 0].max():.2f} km",
        flush=True,
    )
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    arms = ARMS if not quick else ("oracle", "physics", "pinn", "nn")
    elev_arms = ELEV_ARMS if not quick else arms
    itr, _ = elevation_split(prob)
    t0 = time.time()

    if "main" in parts:
        stations = load_field(case).frame()
        stations["split"] = "train"
        stations.loc[elevation_split(prob)[1], "split"] = "test"
        save_table(stations, tr, "stations")

        ex = study_elevation(prob, seeds=seeds, arms=elev_arms)
        # the protocol's extrapolation study on the height split, under the
        # name the cross-track tables read
        save_table(ex, tr, "extrapolation")
        print(f"   elevation split done ({time.time() - t0:.0f} s)", flush=True)

        kr = kriging_study(prob, blocks=not quick)
        save_table(kr, tr, "kriging")
        print(f"   kriging done ({time.time() - t0:.0f} s)", flush=True)

        # the derivative behind `dTdz`, checked for convergence in step
        steps = []
        for arm in ("physics", "nn"):
            steps += lapse_step_study(fit_arm(arm, prob, itr, seeds[0]), prob.x)
        steps += lapse_step_study(fit_kriging(prob, itr, "gp"), prob.x)
        save_table(pd.DataFrame(steps), tr, "lapse_step_convergence")

        meta["headline"] = headline_fits(prob, seeds[0])
        meta["splits"] = {
            "elevation_train_frac": ELEV_TRAIN_FRAC,
            "elevation_z_max_train_km": float(prob.x[itr, 0].max()),
            "n_blocks": N_BLOCKS,
        }
        meta["nn_cfg"] = NN_CFG[case]
        meta["pinn_epochs"] = PINN_EPOCHS
        save_json(meta, tr, "meta")

    if "blocks" in parts and not quick:
        bl = study_blocks(prob, seeds=seeds, arms=arms)
        save_table(bl, tr, "blocks")
        print(f"   block hold-out done ({time.time() - t0:.0f} s)", flush=True)

    if "sweeps" in parts and not quick and case == SWEEP_CASE:
        # Budget and noise sweeps on one case, without symbolic regression:
        # a compute limit, stated in the doc. SR is run on both splits.
        sweep_arms = ("oracle", "physics", "pinn", "nn")
        save_table(
            sweep_budget(prob, [25, 50, 100], arms=sweep_arms, progress=False),
            tr,
            "sweep_budget",
        )
        save_table(
            sweep_noise(prob, [0.0, 0.1], arms=sweep_arms, progress=False),
            tr,
            "sweep_noise",
        )
        print(f"   sweeps done ({time.time() - t0:.0f} s)", flush=True)
    return meta


def run(quick: bool = False, cases=tuple(CASES), sim: bool = True) -> dict:
    """Both real cases, the simulated control, the figures and the doc."""
    from . import weather_plots, weather_sim
    from .weather_doc import render_doc

    out: dict[str, Any] = {c: run_case(c, quick=quick) for c in cases}
    if sim:
        out["sim"] = weather_sim.run(quick=quick)
    weather_plots.all_figures(cases)
    render_doc()
    return out
