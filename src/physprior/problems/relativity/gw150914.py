"""Track G -- gravity, strong field. LIGO GW150914.

DATA     Seven (t, f) pairs: the instantaneous gravitational-wave frequency
         during the last 40 ms of the inspiral, measured MODEL-FREE from
         sub-sample zero crossings of the whitened, coherently combined
         H1 + L1 strain. Seven points, per-cycle SNR 2-8. This is as small
         and as noisy as real data gets, and it is not a choice: GW150914's
         SNR of 24 is accumulated coherently over the waveform, so a
         model-free frequency exists only for the last handful of cycles.

LAW      The post-Newtonian inspiral,

             df/dt = (96/5) pi^(8/3) (G Mc/c^3)^(5/3) f^(11/3) * [1 + PN...]

PARAM    Mc, the redshifted (detector-frame) chirp mass. GWTC-1 gives
         28.6 Msun in the source frame at z = 0.09, so 31.17 Msun here.
         A waveform's frequency evolution fixes only this combination of the
         two masses -- nothing in these data can separate m1 from m2.

WHY THIS TRACK   With seven noisy points no black box can say anything. The
         physics arms return a mass. And the PN ORDER of the law that goes
         into the loss decides whether that mass is right: the Newtonian law
         is biased by +9 Msun, and adding the tail and 2PN terms removes the
         bias entirely. That is the clearest statement in this project of
         what "how much physics is in the loss" is worth.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from physprior.benchmark.protocol import (
    Problem,
    law,
    study_extrapolation,
    sweep_budget,
    sweep_noise,
    sweep_physics_weight,
)
from physprior.constants import GW150914_MCHIRP_DETECTOR, T_SUN_S
from physprior.data.sources import gwosc as gw
from physprior.io import save_json, save_table
from physprior.methods.pinn import DEFAULT_PINN, PhysParam, fit_pinn_ode
from physprior.util import first_column

TRACK = "relativity/gw150914"

# GWTC-1 component masses (source frame) -> symmetric mass ratio. This enters
# only the PN CORRECTION terms; the leading order depends on Mc alone.
M1_SOURCE, M2_SOURCE = 35.6, 30.6
ETA = M1_SOURCE * M2_SOURCE / (M1_SOURCE + M2_SOURCE) ** 2
DEFAULT_PN = 3  # 0=Newtonian, 1=1PN, 2=+1.5PN tail, 3=2PN
F_GRID = np.geomspace(20.0, 400.0, 4000)


def fdot_np(f, Mc, eta=ETA, order=DEFAULT_PN):
    """df/dt in Hz/s. `Mc` in solar masses, `f` in Hz."""
    f = np.asarray(f, float)
    M = Mc / eta**0.6 * T_SUN_S  # total mass, in seconds
    v = (np.pi * M * f) ** (1.0 / 3.0)
    br = np.ones_like(f)
    if order >= 1:
        br = br - (743.0 / 336.0 + 11.0 * eta / 4.0) * v**2
    if order >= 2:
        br = br + 4.0 * np.pi * v**3
    if order >= 3:
        br = (
            br
            + (34103.0 / 18144.0 + 13661.0 * eta / 2016.0 + 59.0 * eta**2 / 18.0) * v**4
        )
    return (
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * (Mc * T_SUN_S) ** (5.0 / 3.0)
        * f ** (11.0 / 3.0)
        * np.clip(br, 1e-3, None)
    )


def fdot_t(f, Mc, eta=ETA, order=DEFAULT_PN):
    """The same, in torch, for the ODE residual."""
    M = Mc / eta**0.6 * T_SUN_S
    v = (np.pi * M * f) ** (1.0 / 3.0)
    br = torch.ones_like(f)
    if order >= 1:
        br = br - (743.0 / 336.0 + 11.0 * eta / 4.0) * v**2
    if order >= 2:
        br = br + 4.0 * np.pi * v**3
    if order >= 3:
        br = (
            br
            + (34103.0 / 18144.0 + 13661.0 * eta / 2016.0 + 59.0 * eta**2 / 18.0) * v**4
        )
    return (
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * (Mc * T_SUN_S) ** (5.0 / 3.0)
        * f ** (11.0 / 3.0)
        * torch.clamp(br, min=1e-3)
    )


def f_of_t(t, Mc, tc, order=DEFAULT_PN, eta=ETA):
    """Invert dt/df by quadrature. `tc` absorbs the arbitrary grid endpoint."""
    tau = np.concatenate(
        [[0.0], cumulative_trapezoid(1.0 / fdot_np(F_GRID, Mc, eta, order), F_GRID)]
    )
    tau = tau[-1] - tau  # time before f_max
    t = first_column(t)
    return np.interp(tc - t, tau[::-1], F_GRID[::-1])


def make_law(order=DEFAULT_PN):
    @law(f"df/dt = (96/5) pi^(8/3) (G Mc/c^3)^(5/3) f^(11/3)  [PN order {order}]")
    def law_np(x, Mc, tc):
        return f_of_t(x, Mc, tc, order=order)

    return law_np


# The options `fit_pinn` grew are switches on the `law + sd_y * NN(x)` loop.
# This track's arm is a different model -- the network IS the solution and the
# ODE residual is the physics term -- so only the ones that wrap a fit rather
# than modify its loop carry over. Ensembling does; the rest would have to be
# implemented against this loop, and until they are the arm reports that they
# did not engage rather than returning a baseline fit under their name.
_ODE_SUPPORTS = ("ensemble",)


def _pinn_ode_once(prob, idx, seed, w_phys):
    x, y = prob.sub(idx)
    t = x[:, 0]

    def residual(tq, fq, dfq, Mc):
        return (dfq - fdot_t(fq, Mc)) / 1000.0  # Hz/s -> O(1)

    f = fit_pinn_ode(
        t,
        y,
        residual,
        [PhysParam("Mc", 30.0, positive=True)],
        w_phys=w_phys,
        seed=seed,
        epochs=6000,
        t_domain=(t.min(), t.max()),
        name="pinn",
    )
    f.expression = (
        getattr(prob.law_np, "expression", "inspiral ODE")
        + "  [as an ODE residual, network = f(t)]"
    )
    return f


def _pinn_ode_arm(prob, idx, seed, w_phys, options=DEFAULT_PINN):
    """Track G's `pinn` is the textbook PINN: the network is f(t) and the
    residual of the inspiral ODE, with Mc trainable, is the physics loss."""
    unsupported = [
        name
        for name, on in (
            ("early_stopping", options.early_stopping),
            ("lbfgs", options.lbfgs),
            ("fourier", bool(options.fourier)),
            ("balance", options.balance),
        )
        if on and name not in _ODE_SUPPORTS
    ]
    if options.ensemble > 1:
        members = [
            _pinn_ode_once(prob, idx, seed + k, w_phys) for k in range(options.ensemble)
        ]
        first = members[0]
        keys = first.params.keys()
        first.params = {k: float(np.mean([m.params[k] for m in members])) for k in keys}
        first.param_sigma = {
            k: float(np.std([m.params[k] for m in members], ddof=1)) for k in keys
        }
        preds = [m.predict for m in members]
        first.predict = lambda xq, _p=preds: np.mean([f(xq) for f in _p], axis=0)
        first.extra["ensemble"] = options.ensemble
        fit = first
    else:
        fit = _pinn_ode_once(prob, idx, seed, w_phys)
    fit.extra["options"] = options.tag
    fit.extra["engaged"] = not unsupported
    if unsupported:
        fit.extra["unsupported_options"] = unsupported
    return fit


def problem(order: int = DEFAULT_PN) -> tuple[Problem, dict]:
    tr = gw.frequency_track()
    law_np = make_law(order)

    # The oracle gets the published chirp mass. `tc` is not a published
    # constant -- it is an event-specific nuisance -- so it is fitted with Mc
    # pinned, and the oracle still has zero free PHYSICS parameters.
    from scipy.optimize import curve_fit

    tc0, _ = curve_fit(
        lambda xx, tc: law_np(xx, GW150914_MCHIRP_DETECTOR, tc),
        tr.t_s.reshape(-1, 1),
        tr.f_hz,
        p0=[tr.t_peak_s],
        bounds=([0.0], [0.08]),
        maxfev=40000,
    )

    prob = Problem(
        track=TRACK,
        x=tr.t_s.reshape(-1, 1),
        y=tr.f_hz,
        law_np=law_np,
        law_t=None,
        params=[
            PhysParam("Mc", 30.0, positive=True, lo=5.0, hi=200.0),
            PhysParam("tc", tr.t_peak_s, positive=False, lo=0.0, hi=0.08),
        ],
        theta_published={"Mc": GW150914_MCHIRP_DETECTOR, "tc": float(tc0[0])},
        xlabel="t - t$_{GPS}$  [s]",
        ylabel="GW frequency  [Hz]",
        sr_transform=_sr_transform_factory(tr.t_peak_s),
        sr_kwargs=dict(
            feature_names=["tau"],
            niterations=80,
            maxsize=12,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=[],
        ),
        nn_cfg=dict(width=16, depth=2, weight_decay=1e-3, epochs=6000),
        arm_impl={"pinn": _pinn_ode_arm},
        notes="GW150914 model-free chirp track, H1+L1",
    )
    meta = {
        "provenance": tr.provenance,
        "n_cycles": len(tr),
        "pn_order": order,
        "eta": ETA,
        "t_peak_s": tr.t_peak_s,
        "t_ms": (tr.t_s * 1e3).tolist(),
        "f_hz": tr.f_hz.tolist(),
        "snr": tr.snr.tolist(),
        "published_Mc_detector": GW150914_MCHIRP_DETECTOR,
        "published_Mc_source": GW150914_MCHIRP_DETECTOR / 1.09,
        "oracle_tc_s": float(tc0[0]),
    }
    return prob, meta


def _sr_transform_factory(t_peak: float):
    """SR sees tau = t_peak - t, the time before merger. `t_peak` is the
    envelope peak -- a model-free measurement, available to every arm."""

    def tf(x, y):
        tau = t_peak - first_column(x).reshape(-1, 1)

        def back(xq, raw_predict):
            return raw_predict(t_peak - first_column(xq).reshape(-1, 1))

        return tau, np.asarray(y, float), back

    return tf


def pn_ablation(prob_x, prob_y, orders=(0, 1, 2, 3)) -> list[dict]:
    """How much physics is in the loss? Fit the same data at each PN order."""
    from scipy.optimize import curve_fit

    names = {0: "0PN (Newtonian)", 1: "1PN", 2: "1.5PN (+tail)", 3: "2PN"}
    rows = []
    for o in orders:
        law = make_law(o)
        try:
            popt, pcov = curve_fit(
                lambda xx, Mc, tc, _law=law: _law(xx, Mc, tc),
                prob_x,
                prob_y,
                p0=[31.0, 0.022],
                bounds=([5.0, 0.0], [200.0, 0.08]),
                maxfev=80000,
            )
            err = np.sqrt(np.diag(pcov))
            pred = law(prob_x, *popt)
            # A fit pinned to its bound has not converged, whatever curve_fit
            # returns. The 1PN series is non-monotone -- its bracket can go
            # negative in band -- and this is where that shows up.
            lo, hi = np.array([5.0, 0.0]), np.array([200.0, 0.08])
            pinned = bool(
                np.any(np.abs(popt - lo) < 1e-9) or np.any(np.abs(popt - hi) < 1e-9)
            )
            rows.append(
                {
                    "pn_order": o,
                    "label": names[o],
                    "Mc": float(popt[0]),
                    "Mc_sigma": float(err[0]),
                    "tc_ms": float(popt[1] * 1e3),
                    "bias_Msun": float(popt[0] - GW150914_MCHIRP_DETECTOR),
                    "bias_sigma": float((popt[0] - GW150914_MCHIRP_DETECTOR) / err[0]),
                    "rmse_hz": float(np.sqrt(np.mean((pred - prob_y) ** 2))),
                    "converged": not pinned,
                    "note": "parameter pinned to bound" if pinned else "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "pn_order": o,
                    "label": names[o],
                    "converged": False,
                    "note": str(e)[:120],
                }
            )
    return rows


def sr_exponent(fit, tau_range) -> dict:
    """The inspiral says f ~ tau^(-3/8). What exponent did SR actually find?"""
    from physprior.methods.symbolic import power_law_exponent

    p = power_law_exponent(fit, "tau", tau_range, tol=0.08)
    return {
        "expression": fit.expression,
        "exponent": p,
        "newtonian_exponent": -3.0 / 8.0,
        "deviation": None if p is None else p + 3.0 / 8.0,
    }


def run(quick: bool = False) -> dict:
    prob, meta = problem()
    print(
        f"[{TRACK}] {len(prob)} model-free cycles, "
        f"f = {prob.y.min():.0f}-{prob.y.max():.0f} Hz"
    )

    meta["pn_ablation"] = pn_ablation(prob.x, prob.y)
    # The calibrated SNR cut leaves 6 model-free cycles and 2 are held out,
    # so 4 is the largest meaningful budget; asking for more clamps to
    # duplicates and evaluates the same split twice.
    budgets = [2, 3, 4] if not quick else [2, 4]
    noise = [0.0, 0.02, 0.05, 0.10, 0.20] if not quick else [0.0, 0.1]

    save_table(sweep_budget(prob, budgets), TRACK, "sweep_budget")
    save_table(sweep_noise(prob, noise, n_train=5), TRACK, "sweep_noise")
    save_table(study_extrapolation(prob, train_frac=0.6), TRACK, "extrapolation")
    # Here w_phys weights the ODE RESIDUAL rather than a correction penalty,
    # so the dial means something different from the other three tracks: it
    # is how hard the network is pushed to satisfy the inspiral equation.
    weights = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3] if not quick else [0.0, 1.0]
    save_table(
        sweep_physics_weight(prob, weights, n_train=5), TRACK, "sweep_physics_weight"
    )

    from physprior.benchmark.protocol import fit_arm

    allidx = np.arange(len(prob))
    head = {}
    for arm in ("physics", "pinn", "sr", "nn"):
        f = fit_arm(arm, prob, allidx, seed=11)
        head[arm] = {
            "params": f.params,
            "sigma": f.param_sigma,
            "expression": f.expression,
            "seconds": f.seconds,
        }
        if arm == "sr":
            tau = meta["t_peak_s"] - prob.x[:, 0]
            head[arm]["law_check"] = sr_exponent(f, (tau.min(), tau.max()))
    meta["headline"] = head
    save_json(meta, TRACK, "meta")
    return meta
