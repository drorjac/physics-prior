"""Run the gravity problem: the real-data track, the simulations, the law
recovery from them, and the animations.

    physprior run gravity
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.constants import AU_M, G_NEWTON
from physprior.io import save_json, save_table
from physprior.units import require_not_none
from physprior.viz import plots as P
from physprior.viz.animate import animate_orbits, animate_training

from . import discovery, kepler, orbits

PROBLEM = "gravity"


def _fig(name):
    return get_settings().figures(PROBLEM) / name


def simulations(quick: bool = False) -> dict:
    out: dict = {}
    print(f"[{PROBLEM}] simulations", flush=True)

    # 1. symplectic vs RK4 -- the integrator IS part of the physics model
    dts = (1.0, 2.0, 4.0) if quick else (0.25, 1.0, 2.0, 4.0)
    rows = orbits.integrator_comparison(dt_days=dts, years=50.0 if quick else 200.0)
    save_table(pd.DataFrame(rows), PROBLEM, "integrator_comparison")
    out["integrator_comparison"] = rows
    print("   integrator comparison done", flush=True)

    # 2. two-body orbit
    tb = orbits.two_body("Mercury", n_orbits=3.0, steps_per_orbit=2000, stride=8)
    out["two_body_drift"] = tb.drift(G_NEWTON)
    animate_orbits(
        tb,
        _fig("two_body_mercury.gif"),
        scale=AU_M,
        title="Sun + Mercury (velocity Verlet)",
        frames=110,
        sizes=[13, 7],
    )

    # 3. three-body: the periodic choreography and its chaotic twin
    f8 = orbits.three_body("figure8", n_periods=3.0, steps_per_period=4000, stride=8)
    animate_orbits(
        f8,
        _fig("three_body_figure8.gif"),
        scale=1.0,
        unit="natural",
        title="Three-body figure-eight choreography",
        frames=120,
    )
    ch = orbits.three_body(
        "chaotic", n_periods=8.0, steps_per_period=4000, perturbation=1e-3, stride=16
    )
    # 90 frames rather than 120: the artefact then clears the repository's
    # 1 MB large-file gate, which is better kept strict than relaxed for one
    # animation.
    animate_orbits(
        ch,
        _fig("three_body_chaotic.gif"),
        scale=1.0,
        unit="natural",
        title="The same, perturbed by 1e-3",
        frames=90,
    )
    out["figure8_drift"] = f8.drift(1.0)

    ly = orbits.lyapunov_separation(1e-9, n_periods=12.0)
    out["lyapunov"] = {k: v for k, v in ly.items() if k not in ("t", "separation")}
    save_table(
        pd.DataFrame({"t": ly["t"], "separation": ly["separation"]}),
        PROBLEM,
        "lyapunov_separation",
    )
    _plot_lyapunov(ly)
    print("   three-body + Lyapunov done", flush=True)

    # 4. the real solar system, from real initial conditions
    ss = orbits.solar_system(years=12.0, dt_days=2.0, stride=30)
    out["solar_system_drift"] = ss.drift(G_NEWTON)
    animate_orbits(
        ss,
        _fig("solar_system_inner.gif"),
        scale=AU_M,
        title="Inner solar system from JPL initial conditions",
        frames=120,
        sizes=[13, 6, 7, 7, 6, 9, 8, 7, 7],
    )
    print("   solar system done", flush=True)
    return out


def _plot_lyapunov(ly):
    import matplotlib.pyplot as plt

    P.use_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.semilogy(
        ly["t"] / ly["t_period"], ly["separation"], color=P.ARM_COLOR["sr"], lw=2.0
    )
    if ly["lambda"] and np.isfinite(ly["lambda"]):
        ax.semilogy(
            ly["t"] / ly["t_period"],
            ly["perturbation"] * np.exp(ly["lambda"] * ly["t"]),
            "--",
            color=P.INK_MUTED,
            lw=1.6,
            label=f"exp({ly['lambda']:.3f} t)",
        )
        ax.legend(fontsize=9, labelcolor=P.INK_2)
    P._style(
        ax,
        "Two figure-eights differing by 1 part in 10^9",
        "time  [periods]",
        "separation  [natural units]",
    )
    P.save(fig, PROBLEM, "lyapunov")


def run_discovery(quick: bool = False) -> dict:
    print(f"[{PROBLEM}] law discovery from the simulations", flush=True)
    out = {
        "force_law": discovery.force_law(
            "Mercury", n_orbits=2.0, steps_per_orbit=2000 if quick else 4000
        )
    }
    print("   force law done", flush=True)
    out["kepler_law"] = discovery.kepler_law(steps_per_orbit=1500 if quick else 3000)
    save_table(
        pd.DataFrame(out["kepler_law"].pop("rows")), PROBLEM, "kepler_from_simulation"
    )
    print("   Kepler from simulation done", flush=True)
    out["chaotic_law"] = discovery.chaotic_law(
        n_periods=4.0 if quick else 6.0, steps_per_period=4000 if quick else 6000
    )
    print("   chaotic law done", flush=True)
    return out


def training_movie() -> dict:
    """Watch a PINN learn an orbit, with GM as a trainable constant.

    The network is fitted to a noisy radius-versus-time series from the
    simulated orbit; the physics term holds it to the Kepler solution. The
    right-hand panel is the point: GM walking to the value that was used to
    generate the data.
    """
    import torch

    from physprior.methods.pinn import PhysParam, fit_pinn

    print(f"[{PROBLEM}] training movie", flush=True)
    tb = orbits.two_body("Mercury", n_orbits=1.0, steps_per_orbit=600, stride=6)
    t = tb.t
    r = np.linalg.norm(tb.r[:, 1] - tb.r[:, 0], axis=1)
    rng = np.random.default_rng(11)
    r_noisy = r + rng.normal(0.0, 0.02 * np.ptp(r), len(r))

    a, e = tb.meta["a_m"], tb.meta["ecc"]

    def law_np(x, GM):
        tt = x[:, 0] if np.ndim(x) > 1 else np.asarray(x)
        n = np.sqrt(GM / a**3)
        M = n * tt
        E = M.copy()
        for _ in range(60):  # Kepler's equation, Newton
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        return a * (1 - e * np.cos(E))

    def law_t(x, GM):
        tt = x[:, 0] if x.ndim > 1 else x
        n = torch.sqrt(GM / a**3)
        M = n * tt
        E = M.clone()
        for _ in range(60):
            E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
        return a * (1 - e * torch.cos(E))

    gm_true = tb.meta["mu"]
    fit = fit_pinn(
        t.reshape(-1, 1),
        r_noisy,
        law_t,
        [PhysParam("GM", gm_true * 0.55, positive=True)],
        w_phys=30.0,
        epochs=2500,
        record_every=25,
        record_grid=np.linspace(t.min(), t.max(), 260),
    )
    hist = require_not_none(
        fit.history, "fit_pinn was asked to record a history and did not"
    )
    grid = np.linspace(t.min(), t.max(), 260)  # the recording grid

    animate_training(
        hist,
        t / 86400.0,
        r_noisy / AU_M,
        predict_at=lambda j, _x: hist["pred"][j] / AU_M,
        x_grid=grid / 86400.0,
        path=_fig("pinn_learning_orbit.gif"),
        title="A PINN learning Mercury's orbit",
        xlabel="t  [days]",
        ylabel="r  [AU]",
        truth=lambda x: np.interp(x, t / 86400.0, r / AU_M),
        param_name="GM",
        param_true=gm_true,
        frames=60,
        fps=12,
    )

    return {
        "GM_true": gm_true,
        "GM_recovered": fit.params["GM"],
        "GM_rel_error_ppm": (fit.params["GM"] - gm_true) / gm_true * 1e6,
        "GM_initial": gm_true * 0.55,
        "epochs": len(hist["epoch"]),
        "noise_frac": 0.02,
        "w_phys": 30.0,
        "history": {k: v for k, v in hist.items() if k != "pred"},
    }


def run(quick: bool = False) -> dict:
    meta: dict[str, Any] = {"problem": PROBLEM}
    meta["kepler_real_data"] = kepler.run(quick=quick)
    meta["simulations"] = simulations(quick=quick)
    meta["discovery"] = run_discovery(quick=quick)
    meta["training"] = training_movie()
    save_json(meta, PROBLEM, "problem_meta")
    return meta
