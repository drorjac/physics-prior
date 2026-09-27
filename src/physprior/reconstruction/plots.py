"""Figures for the reconstruction study, drawn from results/reconstruction/.

Six series share the categorical palette. The set passes the project's
validator on the light surface for all pairs (normal-vision dE >= 16.3) with
the ochre/orange pair at the CVD floor (dE 6.1, protan), which the method
allows only with secondary encoding: every series also has its own marker,
and legends are always present.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from physprior.io import load_table
from physprior.viz import plots as P

from . import fields as F

TRACK = "reconstruction"
COLOR = {
    "physics": P.ARM_COLOR["physics"],
    "pinn": P.ARM_COLOR["pinn"],
    "pigp": "#c93f8e",
    "gp": "#8a6d00",
    "interp": "#eb6834",
    "nn": P.ARM_COLOR["nn"],
}
MARKER = {"physics": "o", "pinn": "D", "pigp": "s", "gp": "^", "interp": "v", "nn": "P"}
# pigp coincides with physics whenever the model is right; dashed and drawn
# underneath so the physics line stays visible
STYLE: dict[str, dict[str, Any]] = {
    "pigp": {"ls": "--", "zorder": 2},
    "physics": {"zorder": 4},
}
LABEL = {
    "physics": "physics (Green's function fit)",
    "pinn": "PINN (PDE residual + data)",
    "pigp": "physics + GP on residual",
    "gp": "GP (RBF, ML-II)",
    "interp": "thin-plate RBF interpolation",
    "nn": "black-box MLP",
}
ORDER = ["physics", "pinn", "pigp", "gp", "interp", "nn"]
DIM_NAME = {1: "1-D rod", 2: "2-D plate", 3: "3-D box"}
CASE_NAME = {"box": "heat in a box (Dirichlet walls)", "free": "free-space potential"}


def _median_band(g: pd.DataFrame, col: str):
    agg = g.groupby("N")[col].agg(["median", "min", "max"]).sort_index()
    return agg.index.to_numpy(), agg["median"].to_numpy(), agg["min"], agg["max"]


def _series(ax, df, col, methods=ORDER):
    for m in methods:
        g = df[df.method == m]
        if g.empty:
            continue
        n, med, lo, hi = _median_band(g, col)
        ax.plot(
            n,
            med,
            color=COLOR[m],
            marker=MARKER[m],
            ms=6,
            lw=1.8,
            label=LABEL[m],
            mec=P.SURFACE,
            mew=1.0,
            **STYLE.get(m, {"zorder": 3}),
        )
        ax.fill_between(n, lo, hi, color=COLOR[m], alpha=0.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def fig_error_vs_n(case: str, col: str = "nrmse"):
    df = load_table(TRACK, "sweep")
    df = df[df.case == case]
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, d in zip(axes, (1, 2, 3), strict=True):
        _series(ax, df[df.d == d], col)
        ax.axhline(0.1, color=P.INK_MUTED, ls="--", lw=1.0)
        what = "field error" if col == "nrmse" else "error outside the sensors"
        where = DIM_NAME[d] if case == "box" else f"{d}-D"
        P._style(ax, f"{where}: {what}", "sensors N", "nRMSE" if d == 1 else None)
    axes[0].legend(fontsize=8, labelcolor=P.INK_2, loc="lower left")
    fig.suptitle(
        f"{CASE_NAME[case]}; median over seeds 11/23/42, band = min-max; "
        "dashed = target 0.1",
        color=P.INK_2,
        fontsize=10,
    )
    return fig


def fig_samples_needed(target: float = 0.1, metric: str = "nrmse"):
    ns = load_table(TRACK, "samples_needed")
    ns = ns[(ns.target == target) & (ns.metric == metric)]
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, case in zip(axes, F.CASES, strict=True):
        s = ns[ns.case == case]
        for m in ORDER:
            g = s[s.method == m].sort_values("d")
            if g.empty:
                continue
            ax.plot(
                g.d,
                g.N_star,
                color=COLOR[m],
                lw=1.8,
                label=LABEL[m],
                marker=MARKER[m],
                **STYLE.get(m, {"zorder": 3}),
            )
            c = g[g.censored]
            if not c.empty:
                ax.scatter(
                    c.d,
                    c.N_star,
                    s=90,
                    facecolor=P.SURFACE,
                    edgecolor=COLOR[m],
                    zorder=4,
                )
                for _, r in c.iterrows():
                    ax.annotate(
                        "",
                        (r.d, r.N_star * 1.9),
                        (r.d, r.N_star * 1.05),
                        arrowprops={"arrowstyle": "->", "color": COLOR[m]},
                    )
        P_d = [F.Geometry(case, d).n_unknowns() for d in (1, 2, 3)]
        ax.plot(
            (1, 2, 3), P_d, color=P.INK_MUTED, ls=":", lw=1.4, label="unknowns P(d)"
        )
        ax.set_yscale("log")
        ax.set_xticks((1, 2, 3))
        P._style(
            ax,
            CASE_NAME[case],
            "dimension d",
            f"sensors to reach nRMSE {target:g}" if case == "box" else None,
        )
    axes[0].legend(fontsize=8, labelcolor=P.INK_2, loc="upper left")
    if ns.censored.any():
        fig.suptitle(
            "open circle with arrow: never reached within the sensor budget "
            "(lower bound)",
            color=P.INK_2,
            fontsize=9,
        )
    return fig


def fig_mismatch():
    df = load_table(TRACK, "mismatch")
    clean = load_table(TRACK, "sweep")
    clean = clean[(clean.case == "box") & (clean.d == 2) & (clean.method == "physics")]
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    titles = {
        "extra_source": "unmodelled fourth source",
        "neumann_wall": "insulated wall modelled as held",
    }
    for ax, kind in zip(axes, titles, strict=True):
        _series(ax, df[df.kind == kind], "nrmse", ["physics", "pigp", "gp", "interp"])
        n, med, _, _ = _median_band(clean, "nrmse")
        ax.plot(
            n, med, color=P.INK_MUTED, ls="--", lw=1.2, label="physics, correct model"
        )
        P._style(ax, f"2-D plate: {titles[kind]}", "sensors N", "nRMSE")
    axes[0].legend(fontsize=8, labelcolor=P.INK_2, loc="lower left")
    return fig


def fig_noise():
    df = load_table(TRACK, "noise")
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, d in zip(axes, (1, 2, 3), strict=True):
        g0 = df[df.d == d]
        for m in ("physics", "gp", "interp"):
            g = g0[g0.method == m].groupby("noise").nrmse.median()
            ax.plot(g.index, g.values, color=COLOR[m], marker=MARKER[m], label=LABEL[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        n = int(g0.N.iloc[0]) if len(g0) else 0
        P._style(ax, f"{DIM_NAME[d]}, N = {n}", "noise / field std", "nRMSE")
    axes[0].legend(fontsize=8, labelcolor=P.INK_2)
    return fig


def fig_convergence():
    s = load_table(TRACK, "convergence_series")
    fs = load_table(TRACK, "convergence_fd_series")
    mm = load_table(TRACK, "convergence_fd_mms")
    fr = load_table(TRACK, "convergence_free")
    P.use_style()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    dim_color = {1: P.ARM_COLOR["physics"], 2: "#eb6834", 3: P.ARM_COLOR["nn"]}
    dim_marker = {1: "o", 2: "s", 3: "^"}
    for d in (1, 2, 3):
        kw = {"color": dim_color[d], "marker": dim_marker[d], "label": f"{d}-D"}
        g = s[s.d == d]
        ax[0].semilogy(g.kmax, np.maximum(g.rel_max_err, 1e-17), **kw)
        g = fs[fs.d == d]
        ax[1].loglog(g.h, g.rel_max_err, **kw)
        g = mm[(mm.d == d) & (mm.boundary == "dirichlet")]
        ax[2].loglog(g.h, g.max_err, **kw)
        g = fr[fr.d == d]
        ax[3].loglog(g.h, g.rel_max_residual, **kw)
    g = mm[mm.boundary == "neumann_x1"]
    ax[2].loglog(
        g.h, g.max_err, color=P.INK_2, ls="--", marker="x", label="2-D, Neumann wall"
    )
    for a in ax[1:]:
        h = np.array(a.get_xlim())
        y0 = a.get_lines()[0].get_ydata()[0]
        x0 = a.get_lines()[0].get_xdata()[0]
        a.plot(h, y0 * (h / x0) ** 2, color=P.INK_MUTED, lw=1, ls=":", label="h^2")
    P._style(ax[0], "sine series vs modes kept", "modes per axis", "max error / std")
    P._style(ax[1], "FD solver vs series", "grid spacing h", "max error / std")
    P._style(ax[2], "FD vs manufactured solution", "grid spacing h", "max error")
    P._style(
        ax[3], "free potentials: -lap(Phi) - g", "stencil h", "max residual / peak"
    )
    for a in ax:
        a.legend(fontsize=8, labelcolor=P.INK_2)
        a.xaxis.set_minor_formatter(NullFormatter())
    return fig


def fig_examples(seed: int = 11):
    """One reconstruction per dimension, from the reporting seed `seed`."""
    import json

    from physprior.config import get_settings

    from .study import NOISE, fit_method, make_scene

    tuned = json.loads((get_settings().results_dir / TRACK / "tuning.json").read_text())
    P.use_style()
    fig = plt.figure(figsize=(14, 7.8))
    gs = fig.add_gridspec(2, 4)
    # 1-D rod
    ax = fig.add_subplot(gs[0, :2])
    sc = make_scene(F.draw_field("box", 1, seed))
    x, y, _ = F.sensors(sc.field, 8, seed, NOISE, sc.scale)
    xs = sc.X[:, 0]
    ax.plot(xs, sc.truth, color=P.INK, lw=1.2, ls="--", label="truth")
    for m in ("physics", "gp", "nn"):
        f = fit_method(m, x, y, sc.field.geom, seed, tuned)
        ax.plot(xs, f.predict(sc.X), color=COLOR[m], label=LABEL[m])
    ax.plot(x[:, 0], y, "o", color=P.INK, ms=5, label="sensors (N = 8)")
    for c in sc.field.centers[:, 0]:
        ax.axvline(c, color=P.GRID, lw=3, zorder=0)
    P._style(ax, "1-D rod, N = 8 (grey bars: true sources)", "x", "temperature u")
    ax.legend(fontsize=8, labelcolor=P.INK_2)
    # free potential 1-D, to show extrapolation
    ax = fig.add_subplot(gs[0, 2:])
    sc = make_scene(F.draw_field("free", 1, seed))
    x, y, _ = F.sensors(sc.field, 16, seed, NOISE, sc.scale)
    xs = sc.X[:, 0]
    ax.axvspan(-0.5, 0.0, color=P.GRID, alpha=0.5, lw=0)
    ax.axvspan(1.0, 1.5, color=P.GRID, alpha=0.5, lw=0)
    ax.plot(xs, sc.truth, color=P.INK, lw=1.2, ls="--", label="truth")
    for m in ("physics", "gp", "nn"):
        f = fit_method(m, x, y, sc.field.geom, seed, tuned)
        ax.plot(xs, f.predict(sc.X), color=COLOR[m], label=LABEL[m])
    ax.plot(x[:, 0], y, "o", color=P.INK, ms=5, label="sensors (N = 16)")
    P._style(ax, "1-D potential; shaded = outside the sensors", "x", "potential")
    # 2-D plate maps
    sc = make_scene(F.draw_field("box", 2, seed))
    n = 32
    x, y, _ = F.sensors(sc.field, n, seed, NOISE, sc.scale)
    m2 = round(float(np.sqrt(len(sc.X))))
    vmin, vmax = sc.truth.min(), sc.truth.max()
    panels = [("truth", sc.truth)]
    for m in ("physics", "gp", "nn"):
        f = fit_method(m, x, y, sc.field.geom, seed, tuned)
        panels.append((LABEL[m].split(" (")[0], f.predict(sc.X)))
    for i, (title, v) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(
            v.reshape(m2, m2).T,
            origin="lower",
            extent=(0, 1, 0, 1),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.plot(x[:, 0], x[:, 1], ".", color="white", ms=4)
        if i == 0:
            ax.plot(*sc.field.centers.T, "x", color="white", ms=8, mew=2)
        ax.grid(False)
        P._style(ax, f"2-D plate, N = {n}: {title}")
    return fig


def make_all(quick: bool = False) -> None:
    for case in F.CASES:
        P.save(fig_error_vs_n(case), TRACK, f"error_vs_n_{case}")
    P.save(fig_error_vs_n("free", "nrmse_out"), TRACK, "extrapolation_free")
    P.save(fig_error_vs_n("box", "nrmse_out"), TRACK, "extrapolation_box")
    P.save(fig_samples_needed(), TRACK, "samples_needed")
    P.save(fig_mismatch(), TRACK, "mismatch")
    P.save(fig_noise(), TRACK, "noise")
    P.save(fig_convergence(), TRACK, "convergence")
    P.save(fig_examples(), TRACK, "examples")
