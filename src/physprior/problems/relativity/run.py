"""Run the relativity problem.

physprior run relativity
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.constants import AU_M, GW150914_MCHIRP_DETECTOR
from physprior.io import save_json, save_table
from physprior.numerics.integrators import Trajectory
from physprior.viz import plots as P
from physprior.viz.animate import animate_orbits

from . import discovery, gw150914, mercury, spacetime

PROBLEM = "relativity"


def _fig(name):
    return get_settings().figures(PROBLEM) / name


def simulations(quick: bool = False) -> dict:
    out: dict = {}
    print(f"[{PROBLEM}] simulations", flush=True)

    out["mercury_precession"] = spacetime.mercury_precession(n_orbits=5)
    out["light_deflection"] = spacetime.light_deflection()

    # Precession is 5e-7 rad per orbit and invisible at true strength, so the
    # animation exaggerates it. The boost is stated on the figure itself.
    boost = 4.0e4
    run = spacetime.schwarzschild_orbit(
        n_orbits=6.0, gr_boost=boost, points_per_orbit=1500
    )
    traj = Trajectory(
        t=run.phi,
        r=np.stack(
            [
                np.zeros((len(run.phi), 3)),
                np.stack([run.x, run.y, np.zeros_like(run.x)], -1),
            ],
            1,
        ),
        v=np.zeros((len(run.phi), 2, 3)),
        masses=np.array([1.0, 0.0]),
        method="DOP853",
        dt=float(run.phi[1] - run.phi[0]),
        names=["Sun", "Mercury"],
    )
    animate_orbits(
        traj,
        _fig("schwarzschild_precession.gif"),
        scale=AU_M,
        title=f"Perihelion precession, GR term x{boost:.0g}",
        frames=120,
        sizes=[13, 7],
    )

    rows = [
        spacetime.schwarzschild_orbit(n_orbits=3.0, gr_boost=b).meta
        | {
            "gr_boost": b,
            "measured_per_orbit_rad": spacetime.schwarzschild_orbit(
                n_orbits=3.0, gr_boost=b
            ).precession_per_orbit,
        }
        for b in ((1.0, 1e3, 1e4) if quick else (1.0, 1e2, 1e3, 1e4, 1e5))
    ]
    out["precession_vs_boost"] = [
        {
            "gr_boost": r["gr_boost"],
            "measured_rad": r["measured_per_orbit_rad"],
            "analytic_rad": r["analytic_precession_per_orbit"],
            "ratio": r["measured_per_orbit_rad"] / r["analytic_precession_per_orbit"],
        }
        for r in rows
    ]
    save_table(pd.DataFrame(out["precession_vs_boost"]), PROBLEM, "precession_vs_boost")
    print("   Schwarzschild orbits done", flush=True)

    out["real_vs_simulated_track"] = discovery.real_vs_simulated_track()
    _plot_real_vs_sim(out["real_vs_simulated_track"])
    return out


def _plot_real_vs_sim(d):
    import matplotlib.pyplot as plt

    P.use_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    # Plot the inspiral only. The waveform carries a ringdown tail at fixed
    # frequency, which is needed so the envelope peaks at the merger but is
    # meaningless on a frequency-versus-time axis.
    st = np.array(d["sim_t_s"])
    sf = np.array(d["sim_f_hz"])
    keep = st <= 0.0
    ax.plot(
        st[keep] * 1e3,
        sf[keep],
        "-",
        color=P.INK_MUTED,
        lw=2.0,
        label=f"simulated 2PN inspiral, $M_c$ = {d['mchirp_msun']:.2f} $M_\\odot$",
    )
    ax.plot(
        np.array(d["real_t_s"]) * 1e3 - d["real_t_peak_s"] * 1e3,
        d["real_f_hz"],
        "o",
        color=P.ARM_COLOR["physics"],
        ms=8,
        label="GW150914, measured model-free",
    )
    ax.set_xlim(-120, 20)
    ax.set_ylim(30, 260)
    P._style(
        ax,
        "Real strain against a simulated waveform -- no fitting",
        "time from merger  [ms]",
        "GW frequency  [Hz]",
    )
    ax.legend(fontsize=9, labelcolor=P.INK_2, loc="upper left")
    P.save(fig, PROBLEM, "real_vs_simulated_chirp")


def run_discovery(quick: bool = False) -> dict:
    print(f"[{PROBLEM}] injection tests (the pipeline's own error budget)", flush=True)
    seeds = discovery.VALIDATION_SEEDS[:4] if quick else discovery.VALIDATION_SEEDS
    out: dict[str, Any] = {"injection": discovery.injection_test(seeds=seeds)}
    save_table(
        pd.DataFrame(out["injection"].pop("trials")), PROBLEM, "injection_trials"
    )
    print("   injection done", flush=True)
    if not quick:
        cal = discovery.threshold_calibration()
        save_table(pd.DataFrame(cal), PROBLEM, "threshold_calibration")
        out["threshold_calibration"] = cal
        print("   threshold calibration done", flush=True)
    out["precession_recovery"] = discovery.precession_recovery()
    return out


def run(quick: bool = False) -> dict:
    meta: dict[str, Any] = {
        "problem": PROBLEM,
        "published_Mc_detector": GW150914_MCHIRP_DETECTOR,
    }
    meta["gw150914_real_data"] = gw150914.run(quick=quick)
    meta["mercury_real_data"] = mercury.run(quick=quick)
    meta["simulations"] = simulations(quick=quick)
    meta["discovery"] = run_discovery(quick=quick)
    save_json(meta, PROBLEM, "problem_meta")
    return meta
