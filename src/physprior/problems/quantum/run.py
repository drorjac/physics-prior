"""Run the quantum problem.

physprior run quantum
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import save_json, save_table
from physprior.viz import plots as P
from physprior.viz.animate import animate_wavefunction

from . import cmb, discovery, hydrogen, schrodinger

PROBLEM = "quantum"


def _fig(name):
    return get_settings().figures(PROBLEM) / name


def simulations(quick: bool = False) -> dict:
    out: dict = {}
    print(f"[{PROBLEM}] simulations", flush=True)

    spectra = {}
    for name, sp in (
        ("infinite_well", schrodinger.infinite_well(n_levels=10)),
        ("harmonic", schrodinger.harmonic_oscillator(n_levels=10)),
        ("hydrogen", schrodinger.hydrogen_radial(n_levels=8)),
    ):
        spectra[name] = {
            "n": sp.n.tolist(),
            "energy": sp.energy.tolist(),
            "exact": None if sp.exact is None else sp.exact.tolist(),
            "max_rel_error": sp.max_rel_error,
            "law": sp.meta["law"],
        }
        _plot_eigenstates(sp, name)
    out["spectra"] = spectra

    out["grid_convergence"] = schrodinger.grid_convergence(
        grids=(500, 1000, 2000) if quick else (500, 1000, 2000, 4000, 8000)
    )
    save_table(pd.DataFrame(out["grid_convergence"]), PROBLEM, "grid_convergence")
    out["box_convergence"] = schrodinger.box_convergence(
        r_maxes=(100.0, 400.0) if quick else (100.0, 200.0, 400.0, 900.0)
    )
    save_table(pd.DataFrame(out["box_convergence"]), PROBLEM, "box_convergence")
    out["r_min_sweep"] = schrodinger.r_min_sweep(
        n_grid=(40000, 80000) if quick else (40000, 80000, 160000)
    )
    save_table(pd.DataFrame(out["r_min_sweep"]), PROBLEM, "r_min_sweep")
    print("   convergence studies done", flush=True)

    wp = schrodinger.wavepacket()
    animate_wavefunction(
        wp["x"],
        wp["psi"],
        wp["t"],
        _fig("tunnelling.gif"),
        potential=wp["V"],
        frames=90,
        title=(
            f"A wavepacket at E = {wp['energy']:.2f} meeting a "
            f"barrier of height {wp['barrier_height']:.2f}"
        ),
        energy=wp["energy"],
    )
    out["tunnelling"] = {k: v for k, v in wp.items() if k not in ("x", "t", "psi", "V")}
    print(
        f"   tunnelling: T = {wp['transmission']:.4f}, "
        f"norm drift {wp['norm_drift']:.2e}",
        flush=True,
    )
    return out


def _plot_eigenstates(sp, name):
    import matplotlib.pyplot as plt

    P.use_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    scale = 0.35 * float(np.median(np.diff(sp.energy)))
    for i in range(min(5, len(sp.n))):
        ax.axhline(sp.energy[i], color=P.GRID, lw=1.0)
        ax.plot(
            sp.x,
            sp.energy[i] + scale * sp.psi[i] / np.abs(sp.psi[i]).max(),
            lw=1.8,
            color=P.ARM_COLOR["pinn"] if i % 2 else P.ARM_COLOR["physics"],
        )
        ax.annotate(
            f"n={sp.n[i]}",
            (sp.x[0], sp.energy[i]),
            fontsize=8,
            color=P.INK_2,
            va="bottom",
        )
    finite = np.isfinite(sp.V)
    lo, hi = ax.get_ylim()
    ax.plot(
        sp.x[finite],
        np.clip(sp.V[finite], lo, hi),
        color=P.INK_MUTED,
        lw=1.4,
        ls="--",
        label="V(x)",
    )
    ax.set_ylim(lo, hi)
    P._style(ax, f"{sp.meta['system']}: {sp.meta['law']}", "x", "energy")
    ax.legend(fontsize=9, labelcolor=P.INK_2)
    P.save(fig, PROBLEM, f"eigenstates_{name}")


def run_discovery(quick: bool = False) -> dict:
    print(f"[{PROBLEM}] law discovery from the solved spectra", flush=True)
    out: dict[str, Any] = {
        "spectrum_laws": [
            discovery.spectrum_law(s, n_levels=10)
            for s in ("well", "harmonic", "hydrogen")
        ]
    }
    save_table(
        pd.DataFrame(
            [
                {k: v for k, v in r.items() if k not in ("energies", "n")}
                for r in out["spectrum_laws"]
            ]
        ),
        PROBLEM,
        "spectrum_laws",
    )
    print("   SR on simulated spectra done", flush=True)
    out["schrodinger_vs_nist"] = discovery.schrodinger_vs_nist(n_max=8 if quick else 10)
    print(
        f"   Schrodinger vs NIST: "
        f"{out['schrodinger_vs_nist']['mean_gap_ppm']:+.1f} ppm",
        flush=True,
    )
    return out


def run(quick: bool = False) -> dict:
    meta: dict[str, Any] = {"problem": PROBLEM}
    meta["hydrogen_real_data"] = hydrogen.run(quick=quick)
    meta["cmb_real_data"] = cmb.run(quick=quick)
    meta["simulations"] = simulations(quick=quick)
    meta["discovery"] = run_discovery(quick=quick)
    save_json(meta, PROBLEM, "problem_meta")
    return meta
