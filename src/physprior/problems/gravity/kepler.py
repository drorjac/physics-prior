"""Kepler's third law, from the real JPL DE441 ephemeris.

DATA   semi-major axis `a` and sidereal period `P` for the eight planets at a
       common epoch, from JPL Horizons.

LAW    P = 2 pi sqrt(a^3 / GM)

PARAM  GM_sun.

Eight points is the whole dataset -- there are only eight planets. The
extrapolation test trains on the four terrestrial planets and predicts the
four giants, a factor 20 in `a`.

The companion experiment is `physprior.problems.gravity.discovery`, which recovers the
same 3/2 from SIMULATED orbits where the answer is known exactly. Read the two
together: the simulation gives the method's own noise floor, the ephemeris
gives its performance on real measurements.
"""

from __future__ import annotations

import numpy as np
import torch

from physprior.benchmark.protocol import (
    Problem,
    law,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import (
    AU_M,
    DAY_S,
    GM_SUN,
)
from physprior.data.sources import horizons as eph
from physprior.io import save_json, save_table
from physprior.methods.pinn import PhysParam
from physprior.util import first_column

TRACK = "gravity/kepler"

# The `pinn` arm's physics weight, chosen on the tuning seeds by
# `physprior.benchmark.pinn_tuning` (results/gravity/kepler/tune/w_phys_selection.csv).
PINN_W_PHYS = 100


@law("P = 2*pi*sqrt(a**3/(G*M))")
def law_np(x, GM):
    a = first_column(x) * AU_M
    return 2.0 * np.pi * np.sqrt(a**3 / GM) / DAY_S


def law_t(x, GM):
    a = (x[:, 0] if x.ndim > 1 else x) * AU_M
    return 2.0 * np.pi * torch.sqrt(a**3 / GM) / DAY_S


def _sr_transform(x, y):
    X = first_column(x).reshape(-1, 1)

    def back(xq, raw_predict):
        return raw_predict(first_column(xq).reshape(-1, 1)) * 1000.0

    return X, np.asarray(y, float) / 1000.0, back


def problem() -> tuple[Problem, dict]:
    pt = eph.planets()
    prob = Problem(
        track=TRACK,
        pinn_w_phys=PINN_W_PHYS,
        x=pt.a_au.reshape(-1, 1),
        y=pt.period_d,
        law_np=law_np,
        law_t=law_t,
        params=[PhysParam("GM", 1.0e20, positive=True, lo=1e18, hi=1e22)],
        theta_published={"GM": GM_SUN},
        xlabel="semi-major axis a  [AU]",
        ylabel="sidereal period P  [days]",
        sr_transform=_sr_transform,
        sr_kwargs=dict(
            feature_names=["a"],
            niterations=60,
            maxsize=12,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sqrt"],
        ),
        nn_cfg=dict(width=16, depth=2, weight_decay=1e-4, epochs=6000),
        notes="JPL Horizons DE441 osculating elements, 8 planets, epoch 2020-01-01",
    )
    meta = {
        "planets": pt.name,
        "a_au": pt.a_au.tolist(),
        "period_d": pt.period_d.tolist(),
        "ecc": pt.ecc.tolist(),
        "provenance": pt.provenance,
        "published_GM_sun": GM_SUN,
    }
    return prob, meta


def sr_kepler_exponent(fit, a_range) -> dict:
    from physprior.methods.symbolic import power_law_exponent

    p = power_law_exponent(fit, "a", a_range, tol=0.05)
    return {
        "expression": fit.expression,
        "exponent": p,
        "kepler_exponent": 1.5,
        "deviation": None if p is None else p - 1.5,
    }


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    print(
        f"[{TRACK}] {len(prob)} planets, "
        f"a = {prob.x[:, 0].min():.3f}-{prob.x[:, 0].max():.1f} AU"
    )

    budgets = [3, 4, 5, 6] if not quick else [4, 6]
    noise = [0.0, 0.001, 0.01, 0.05, 0.15] if not quick else [0.0, 0.05]
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [0.0, 1.0]

    save_table(sweep_budget(prob, budgets), TRACK, "sweep_budget")
    save_table(sweep_noise(prob, noise, n_train=6), TRACK, "sweep_noise")
    save_table(study_extrapolation(prob, train_frac=0.5), TRACK, "extrapolation")
    save_table(
        sweep_physics_weight(prob, weights, n_train=6), TRACK, "sweep_physics_weight"
    )

    from physprior.benchmark.protocol import fit_arm

    allidx = np.arange(len(prob))
    head = {}
    for arm in ("physics", "pinn", "sr"):
        f = fit_arm(arm, prob, allidx, seed=11)
        head[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "expression": f.expression,
            "seconds": f.seconds,
        }
        if arm == "sr":
            head[arm]["law_check"] = sr_kepler_exponent(
                f, (prob.x[:, 0].min(), prob.x[:, 0].max())
            )
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    return meta
