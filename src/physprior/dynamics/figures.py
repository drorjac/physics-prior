"""Figures for the dynamics study.

Arms keep one colour across every figure (colour follows the entity). The
six learned ODE arms take the reference categorical order, validated on the
light surface for adjacent pairs, the pairlist that applies to line charts;
every multi-series figure has a legend, so identity is never colour alone.
The oracle is a neutral dashed reference and the incomplete known physics a
neutral dotted one; neither is a competitor.

    python -m physprior.viz.palette "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300" light
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physprior.viz import plots as P

from . import TRACK
from . import metrics as M
from . import systems as S

ARM_COLOR = {
    "direct": "#2a78d6",
    "residual": "#eb6834",
    "node": "#1baf7a",
    "hnn": "#eda100",
    "hnn_leapfrog": "#e87ba4",
    "closure": "#008300",
    "conv": "#4a3aa7",
    "dense": "#e34948",
    "oracle": P.INK_MUTED,
    "known_only": P.INK_2,
    "physics": P.INK_MUTED,
}
ARM_STYLE = {"oracle": "--", "known_only": ":", "physics": "--"}
ARM_LABEL = {
    "direct": "direct  u' = NN(u)",
    "residual": "residual  u + dt NN(u)",
    "node": "neural ODE (RK4)",
    "hnn": "HNN (RK4)",
    "hnn_leapfrog": "HNN (leapfrog)",
    "closure": "known physics + closure",
    "conv": "local conv (kernel 5)",
    "dense": "dense MLP",
    "oracle": "true field, RK4 at dt",
    "known_only": "known physics only",
    "physics": "correct PDE, coarse FD",
}
ODE_ORDER = [
    "direct",
    "residual",
    "node",
    "hnn",
    "hnn_leapfrog",
    "closure",
    "oracle",
    "known_only",
]
PDE_ORDER = ["conv", "dense", "closure", "physics"]


def _line(ax, x, y, arm, **kw):
    ax.plot(
        x,
        y,
        color=ARM_COLOR[arm],
        ls=ARM_STYLE.get(arm, "-"),
        lw=1.6 if arm in ARM_STYLE else 2.0,
        label=ARM_LABEL[arm],
        **kw,
    )


def _agg_curves(
    curves: pd.DataFrame, system: str, col: str, split="test", exp=("main", "ref")
) -> dict[str, pd.DataFrame]:
    c = curves[
        (curves.system == system)
        & (curves.split == split)
        & curves.exp.isin(exp)
        & (curves.loss == "onestep")
    ]
    out = {}
    for arm, g in c.groupby("arm"):
        if col in g:
            out[arm] = g.groupby("t")[col].median().reset_index()
    return out


def _finite_floor(y, floor=1e-8):
    y = np.asarray(y, float)
    return np.where(np.isfinite(y), np.maximum(y, floor), np.nan)


def fig_error_vs_horizon(curves: pd.DataFrame, split: str = "test"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, name in zip(
        axes.flat, ("pendulum", "duffing", "kepler", "lorenz"), strict=False
    ):
        sys = S.get_system(name)
        agg = _agg_curves(curves, name, "err", split)
        if not agg:
            ax.set_visible(False)
            continue
        for arm in ODE_ORDER:
            if arm in agg:
                _line(ax, agg[arm].t, _finite_floor(agg[arm].err), arm)
        thr = M.LORENZ_THR if sys.chaotic else M.VALID_THR
        ax.axhline(thr, color=P.INK_MUTED, lw=0.8)
        ax.axvline(sys.train_steps * sys.dt, color=P.INK_MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        P._style(ax, f"{name} ({split})", "time", "rollout error [train sd]")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Rollout error vs horizon, median over initial states and seeds "
        "(dotted: end of a training trajectory; solid: validity threshold)",
        fontsize=10,
    )
    return fig


def fig_invariants(curves: pd.DataFrame):
    panels = [
        ("pendulum", "H", "|dH| / (H0 + 1)"),
        ("kepler", "E", "|dE / E0|"),
        ("kepler", "L", "|dL / L0|"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, key, lab) in zip(axes, panels, strict=False):
        agg = _agg_curves(curves, name, f"drift_{key}")
        for arm in ODE_ORDER:
            if arm in agg:
                _line(
                    ax, agg[arm].t, _finite_floor(agg[arm][f"drift_{key}"], 1e-10), arm
                )
        sys = S.get_system(name)
        ax.axvline(sys.train_steps * sys.dt, color=P.INK_MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        P._style(ax, f"{name}: {key}", "time", lab)
    axes[0].legend(fontsize=8, loc="upper left")
    return fig


def fig_lorenz(rows: pd.DataFrame):
    lz = rows[(rows.system == "lorenz") & (rows.exp == "main") & (rows.split == "test")]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arm in ODE_ORDER:
        g = lz[lz.arm == arm]
        if len(g):
            ax.scatter(
                g.onestep,
                g.vpt_lyap,
                s=40,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm],
                edgecolor=P.SURFACE,
                linewidth=1.5,
            )
    ref = rows[(rows.system == "lorenz") & (rows.exp == "ref") & (rows.split == "test")]
    for _, r in ref.iterrows():
        ax.scatter(
            r.onestep,
            r.vpt_lyap,
            s=60,
            marker="D",
            color=ARM_COLOR[r.arm],
            label=ARM_LABEL[r.arm],
            edgecolor=P.SURFACE,
            linewidth=1.5,
        )
    ax.set_xscale("log")
    P._style(
        ax,
        "Lorenz-63: valid prediction time vs one-step error",
        "one-step error [train sd]",
        "valid time [Lyapunov times]",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    return fig


def fig_objective(rows: pd.DataFrame):
    ob = rows[(rows.split == "test") & rows.exp.isin(["main", "objective"])]
    pairs = sorted(
        {
            (s, a)
            for s, a in ob[ob.exp == "objective"][["system", "arm"]].itertuples(
                index=False
            )
        }
    )
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pairs) + 1.5))
    for i, (s, a) in enumerate(pairs):
        one = ob[(ob.system == s) & (ob.arm == a) & (ob.exp == "main")].valid_over_train
        unr = ob[
            (ob.system == s) & (ob.arm == a) & (ob.exp == "objective")
        ].valid_over_train
        one, unr = np.maximum(one, 1e-3), np.maximum(unr, 1e-3)
        m1, m2 = float(np.median(one)), float(np.median(unr))
        ax.plot([m1, m2], [i, i], color=P.GRID, lw=3, zorder=1)
        ax.scatter(one, [i] * len(one), s=18, color=P.INK_MUTED, zorder=2)
        ax.scatter(unr, [i] * len(unr), s=18, color=ARM_COLOR[a], zorder=2)
        ax.scatter(
            [m1],
            [i],
            s=70,
            facecolor=P.SURFACE,
            edgecolor=P.INK_2,
            zorder=3,
            label="one-step loss" if i == 0 else None,
        )
        ax.scatter(
            [m2],
            [i],
            s=70,
            color=ARM_COLOR[a],
            zorder=3,
            edgecolor=P.SURFACE,
            label="unrolled loss" if i == 0 else None,
        )
    ax.set_yticks(
        range(len(pairs)), [f"{s} / {ARM_LABEL[a].split('  ')[0]}" for s, a in pairs]
    )
    ax.set_xscale("log")
    P._style(
        ax,
        "Training objective: valid horizon (median, dots = seeds)",
        "valid horizon / training-trajectory length",
        None,
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def fig_data_efficiency(rows: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, col, lab, order) in zip(
        axes,
        (
            (
                "pendulum",
                "valid_over_train",
                "valid horizon / training length",
                ODE_ORDER,
            ),
            ("burgers", "err_end", "error at end of rollout [initial RMS]", PDE_ORDER),
        ),
        strict=False,
    ):
        d = rows[
            (rows.system == name)
            & (rows.split == "test")
            & rows.exp.isin(["main", "data"])
        ]
        for arm in order:
            g = d[d.arm == arm]
            if not len(g):
                continue
            m = g.groupby("n_traj")[col].median()
            _line(ax, m.index, _finite_floor(m.values, 1e-4), arm, marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        P._style(ax, f"{name}: data efficiency", "training trajectories", lab)
        ax.legend(fontsize=8)
    return fig


def fig_pde_error(curves: pd.DataFrame, split: str = "test"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, name in zip(axes, ("heat", "advection", "burgers"), strict=False):
        agg = _agg_curves(curves, name, "err", split)
        for arm in PDE_ORDER:
            if arm in agg:
                _line(ax, agg[arm].t, _finite_floor(agg[arm].err), arm)
        ax.axhline(M.VALID_THR, color=P.INK_MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        P._style(ax, f"{name} ({split})", "time", "rollout error [initial RMS]")
    axes[-1].legend(fontsize=8)
    return fig


def fig_burgers_rollout(examples: dict, split: str = "test", i: int = 0):
    from physprior.dynamics.pdes import PDES, grid

    arms = [
        a for a in ("conv", "dense", "closure") if ("pde", "burgers", a) in examples
    ]
    if not arms:
        return None
    truth = examples[("pde", "burgers", arms[0])][split]["truth"][i]
    panels = [("truth", truth)] + [
        (ARM_LABEL[a], examples[("pde", "burgers", a)][split]["pred"][i]) for a in arms
    ]
    lim = float(np.nanmax(np.abs(truth)))
    t = np.arange(truth.shape[0]) * PDES["burgers"].dt
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels), 4), sharey=True
    )
    for ax, (lab, u) in zip(axes, panels, strict=False):
        u = np.where(np.isfinite(u), u, np.nan)
        im = ax.pcolormesh(
            grid(), t, u, cmap=P._diverging_cmap(), vmin=-lim, vmax=lim, shading="auto"
        )
        ax.grid(False)
        P._style(ax, lab, "x", "t" if ax is axes[0] else None)
    fig.colorbar(im, ax=axes, shrink=0.8, label="u")
    return fig


def fig_pendulum_phase(examples: dict, i: int = 0):
    arms = [a for a in ODE_ORDER if ("ode", "pendulum", a) in examples]
    if not arms:
        return None
    fig, axes = plt.subplots(
        1, len(arms), figsize=(2.9 * len(arms), 3.2), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms, strict=False):
        ex = examples[("ode", "pendulum", a)]["test"]
        tr, pr = ex["truth"][i], ex["pred"][i]
        ax.plot(tr[:, 0], tr[:, 1], color=P.GRID, lw=3)
        ax.plot(pr[:, 0], pr[:, 1], color=ARM_COLOR[a], lw=0.8)
        P._style(ax, ARM_LABEL[a].split("  ")[0], "q", "p" if ax is axes[0] else None)
    return fig


def regenerate() -> list:
    """Redraw every figure from results/dynamics without training anything."""
    from physprior.io import load_table

    from .study import load_examples, load_rows

    return make_figures(load_rows(), load_table(TRACK, "curves"), load_examples())


def make_figures(rows: pd.DataFrame, curves: pd.DataFrame, examples: dict) -> list:
    P.use_style()
    out = []
    for name, fn in (
        ("ode_error_vs_horizon", lambda: fig_error_vs_horizon(curves, "test")),
        ("ode_error_vs_horizon_ood", lambda: fig_error_vs_horizon(curves, "ood")),
        ("invariants", lambda: fig_invariants(curves)),
        ("lorenz_valid_time", lambda: fig_lorenz(rows)),
        ("objective", lambda: fig_objective(rows)),
        ("data_efficiency", lambda: fig_data_efficiency(rows)),
        ("pde_error_vs_horizon", lambda: fig_pde_error(curves, "test")),
        ("pde_error_vs_horizon_ood", lambda: fig_pde_error(curves, "ood")),
        ("burgers_rollout", lambda: fig_burgers_rollout(examples)),
        ("pendulum_phase", lambda: fig_pendulum_phase(examples)),
    ):
        fig = fn()
        if fig is not None:
            out.append(P.save(fig, TRACK, name))
    return out
