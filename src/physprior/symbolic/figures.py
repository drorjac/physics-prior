"""Figures for the symbolic-regression study, drawn from results/symbolic/.

    make_figures()  ->  figures/symbolic/{growth,recovery,pareto,vocabulary,sindy}.png

Colours: the searchers take the repo's validated categorical slots (GP blue,
exhaustive aqua, PySR orange -- the slot the `sr` arm uses everywhere else);
an ordered family of vocabularies or generations takes a single-hue ramp;
the exact law is the neutral dashed reference line, as the oracle is in the
rest of the repo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from physprior.io import load_json, load_table
from physprior.viz import plots as P

AREA = "symbolic"
METHOD_COLOR = {"gp": "#2a78d6", "exhaustive": "#1baf7a", "pysr": "#eb6834"}
METHOD_LABEL = {"gp": "GP (this repo)", "exhaustive": "exhaustive", "pysr": "PySR"}
DERIV_COLOR = {"fd": "#eb6834", "savgol": "#2a78d6"}
DERIV_LABEL = {"fd": "central differences", "savgol": "Savitzky-Golay"}
# light -> dark single-hue ramp for ordered families
BLUES = ["#b7d3f2", "#86b4e8", "#5595dd", "#2a78d6", "#1f5aa3", "#153d70", "#0b2240"]


def _ramp(k: int) -> list[str]:
    idx = np.linspace(0, len(BLUES) - 1, k).round().astype(int)
    return [BLUES[i] for i in idx]


def _try(name: str) -> pd.DataFrame | None:
    try:
        return load_table(AREA, name)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------


def fig_growth():
    counts = _try("growth_counts")
    vocab = _try("time_vs_vocabulary")
    pruned = _try("growth_pruned")
    if counts is None:
        return None
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    groups = list(counts.groupby("n_ops", sort=True))
    cols = _ramp(len(groups))
    for (n_ops, g), c in zip(groups, cols, strict=True):
        # with no unary operator, even sizes are impossible: zero, not drawn
        g = g[g["n_trees_raw"] > 0]
        ax.semilogy(
            g["size"], g["n_trees_raw"], color=c, lw=1.6, label=f"{n_ops} operators"
        )
    if pruned is not None:
        big = pruned[pruned.vocabulary == pruned.vocabulary.iloc[-1]]
        ax.semilogy(
            big["size"],
            big["n_trees_pruned"],
            ls="none",
            marker="o",
            mfc="none",
            color=P.INK_2,
            label="9 operators, redundancy removed",
        )
    P._style(ax, "Distinct trees of each size", "tree size (nodes)", "number of trees")
    ax.legend(fontsize=8, ncol=1)
    ax = axes[1]
    if vocab is not None:
        ax.semilogy(
            vocab["n_ops"],
            vocab["seconds"],
            marker="o",
            color=METHOD_COLOR["exhaustive"],
        )
        for _, r in vocab.iterrows():
            ax.annotate(
                f"{int(r.n_evaluated):,}",
                (r.n_ops, r.seconds),
                fontsize=7,
                color=P.INK_2,
                xytext=(4, -10),
                textcoords="offset points",
            )
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    P._style(
        ax,
        "Exhaustive search for Bohr's law (labels: trees fitted)",
        "operators in the vocabulary",
        "seconds to reach the law",
    )
    return P.save(fig, AREA, "growth")


def fig_recovery():
    frames = [
        d for d in (_try("recovery_noise"), _try("recovery_pysr")) if d is not None
    ]
    budget = _try("recovery_budget")
    if not frames:
        return None
    noise = pd.concat(frames, ignore_index=True)
    if budget is not None:
        # n = 24 at 1 % noise is the noise sweep's run, not repeated
        extra = noise[(noise.noise == 0.01) & (noise.method != "pysr")]
        budget = pd.concat([budget, extra], ignore_index=True)
    laws = list(dict.fromkeys(noise["law"]))
    P.use_style()
    fig, axes = plt.subplots(2, len(laws), figsize=(3.0 * len(laws), 5.6), sharey=True)
    axes = np.atleast_2d(axes)
    for j, law in enumerate(laws):
        for i, (df, col, xlabel) in enumerate(
            ((noise, "noise", "relative noise"), (budget, "n", "samples"))
        ):
            ax = axes[i, j]
            if df is None:
                continue
            d = df[df.law == law]
            for m in ("exhaustive", "gp", "pysr"):
                dm = d[d.method == m]
                if dm.empty:
                    continue
                g = dm.groupby(col).agg(
                    rate=("recovered", "mean"), front=("on_front", "mean")
                )
                xs = np.asarray(g.index, float)
                if col == "noise":
                    xs = np.where(xs == 0, 3e-4, xs)
                ax.plot(
                    xs,
                    g["rate"],
                    marker="o",
                    color=METHOD_COLOR[m],
                    label=METHOD_LABEL[m],
                )
                ax.plot(xs, g["front"], ls=":", lw=1.2, color=METHOD_COLOR[m])
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            if col == "n":
                ax.xaxis.set_major_locator(FixedLocator(sorted(d[col].unique())))
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylim(-0.05, 1.05)
            P._style(
                ax,
                law.replace("_", " ") if i == 0 else None,
                xlabel,
                "recovered (of 3 seeds)" if j == 0 else None,
            )
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.text(
        0.5,
        -0.02,
        "solid: the selected expression; dotted: the law is somewhere on "
        "the front. Noise 0 plotted at 3e-4.",
        ha="center",
        fontsize=8,
        color=P.INK_2,
    )
    return P.save(fig, AREA, "recovery")


def fig_pareto():
    pf = _try("pareto")
    evo = _try("pareto_gp_generations")
    if pf is None:
        return None
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for m in ("exhaustive", "gp", "pysr"):
        d = pf[pf.method == m].sort_values("complexity")
        if d.empty:
            continue
        ax.step(
            d["complexity"],
            d["loss"],
            where="post",
            color=METHOD_COLOR[m],
            label=METHOD_LABEL[m],
            marker="o",
            ms=4,
        )
        s = d[d.selected.astype(bool)]
        ax.plot(
            s["complexity"],
            s["loss"],
            ls="none",
            marker="o",
            ms=12,
            mfc="none",
            mec=METHOD_COLOR[m],
            mew=1.8,
        )
    ax.set_yscale("log")
    P._style(
        ax,
        "Loss vs complexity; ring = the 'best' choice",
        "complexity (nodes)",
        "mean squared relative error",
    )
    ax.legend(fontsize=8)
    ax = axes[1]
    if evo is not None:
        gens = sorted(evo["generation"].unique())
        pick = sorted(
            {
                gens[0],
                gens[min(2, len(gens) - 1)],
                gens[min(5, len(gens) - 1)],
                gens[min(10, len(gens) - 1)],
                gens[min(20, len(gens) - 1)],
                gens[-1],
            }
        )
        for g, c in zip(pick, _ramp(len(pick)), strict=True):
            d = evo[evo.generation == g].sort_values("complexity")
            ax.step(
                d["complexity"],
                d["loss"],
                where="post",
                color=c,
                label=f"generation {g}",
            )
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    P._style(ax, "The GP's front, generation by generation", "complexity (nodes)", None)
    return P.save(fig, AREA, "pareto")


def fig_vocabulary():
    import sympy

    from .studies import LAWS

    try:
        cur = load_json(AREA, "vocabulary_curves")
    except FileNotFoundError:
        return None
    law = LAWS["planck"]
    xs = np.geomspace(0.05, 15.0, 400)
    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    seed = min(int(k.split("|")[2]) for k in cur["expressions"])
    for ax, label in zip(axes, ("without exp", "with exp"), strict=True):
        ax.axvspan(*cur["band"], color=P.GRID, alpha=0.6, lw=0)
        ax.loglog(
            xs,
            law.truth(xs),
            ls="--",
            color=P.INK_MUTED,
            lw=1.6,
            label="Planck (exact)",
        )
        for m in ("exhaustive", "gp", "pysr"):
            e = cur["expressions"].get(f"{m}|{label}|{seed}")
            if not e:
                continue
            try:
                f = sympy.lambdify(sympy.Symbol("x0"), sympy.sympify(e), "numpy")
                with np.errstate(all="ignore"):
                    v = np.broadcast_to(np.asarray(f(xs), float), xs.shape).copy()
                v[~np.isfinite(v) | (v <= 0)] = np.nan
            except Exception:
                continue
            ax.loglog(xs, v, color=METHOD_COLOR[m], lw=1.6, label=METHOD_LABEL[m])
        ax.set_ylim(1e-4, 10)
        P._style(
            ax,
            f"Vocabulary {label} (seed {seed})",
            "x = h nu / k T",
            "B(x)" if label == "without exp" else None,
        )
    axes[0].legend(fontsize=8, loc="lower center")
    fig.text(
        0.5,
        -0.02,
        "grey band: where the data are",
        ha="center",
        fontsize=8,
        color=P.INK_2,
    )
    return P.save(fig, AREA, "vocabulary")


def fig_sindy():
    conv = _try("sindy_derivative_convergence")
    amp = _try("sindy_noise_amplification")
    rec = _try("sindy_recovery")
    if conv is None:
        return None
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    ax = axes[0]
    ax.loglog(
        conv["dt"],
        conv["fd_max_err"],
        marker="o",
        color=DERIV_COLOR["fd"],
        label=DERIV_LABEL["fd"],
    )
    if "savgol_max_err" in conv:
        ax.loglog(
            conv["dt"],
            conv["savgol_max_err"],
            marker="o",
            color=DERIV_COLOR["savgol"],
            label=DERIV_LABEL["savgol"] + " (fixed 0.4 s window)",
        )
    ref = conv["fd_max_err"].iloc[0] * (conv["dt"] / conv["dt"].iloc[0]) ** 2
    ax.loglog(conv["dt"], ref, ls="--", color=P.INK_MUTED, lw=1.2, label="slope 2")
    ax.xaxis.set_minor_formatter(NullFormatter())
    P._style(ax, "Clean signal: error vs step", "dt", "max |error| of dx/dt")
    ax.legend(fontsize=8)
    ax = axes[1]
    if amp is not None:
        for m in ("fd", "savgol"):
            g = amp[amp.method == m].groupby("sigma")["rms_err"].mean()
            ax.loglog(
                g.index,
                g.values,
                marker="o",
                color=DERIV_COLOR[m],
                label=DERIV_LABEL[m],
            )
        th = amp.groupby("sigma")["fd_theory"].first()
        ax.loglog(
            th.index,
            th.values,
            ls="--",
            color=P.INK_MUTED,
            lw=1.2,
            label="sigma / (sqrt(2) dt)",
        )
        ax.legend(fontsize=8)
    P._style(
        ax, "Noisy signal, dt = 0.01", "measurement noise sigma", "rms error of dx/dt"
    )
    ax = axes[2]
    if rec is not None:
        for m in ("fd", "savgol"):
            for sysname, ls in (("damped_oscillator", "-"), ("pendulum", "--")):
                d = rec[(rec.method == m) & (rec.system == sysname)]
                if d.empty:
                    continue
                g = d.groupby("noise")["coef_rel_err"].median()
                xs = np.where(g.index == 0, 3e-5, g.index)
                ax.loglog(
                    xs,
                    g.values,
                    marker="o",
                    ls=ls,
                    color=DERIV_COLOR[m],
                    label=f"{DERIV_LABEL[m]}, {sysname.replace('_', ' ')}",
                )
        ax.legend(fontsize=7)
    P._style(
        ax,
        "SINDy coefficient error",
        "relative noise (0 at 3e-5)",
        "||Xi - Xi_true|| / ||Xi_true||",
    )
    return P.save(fig, AREA, "sindy")


def make_figures() -> list:
    out = []
    for f in (fig_growth, fig_recovery, fig_pareto, fig_vocabulary, fig_sindy):
        p = f()
        if p is not None:
            out.append(p)
    return out
