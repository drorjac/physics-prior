"""Figures.

Colour follows the dataviz method. The four competing arms take categorical
slots validated for ALL pairs on the light surface (blue / orange / aqua /
violet: worst CVD dE 9.2, worst normal-vision dE 16.3, by
`scripts/validate_palette.py`, a Python port of the skill's validator that
reproduces its published reference numbers exactly).

The figures are rasterised with an explicit light surface, so the light-mode
validation is the one that applies however the reader's theme is set. The
blue/violet pair does NOT clear the dark-surface gate, which is precisely why
the surface is pinned rather than left to the viewer.

`oracle` is deliberately NOT a categorical hue. It is not a competitor -- it
is the published law, the ceiling -- so it is drawn as a neutral dashed
reference line. A reference line is not a series and is exempt from the
chroma floor.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from physprior.config import get_settings

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2de"

ARM_COLOR = {
    "physics": "#2a78d6",  # blue
    "sr": "#eb6834",  # orange
    "pinn": "#1baf7a",  # aqua
    "nn": "#4a3aa7",  # violet
    "oracle": INK_MUTED,  # neutral reference, dashed
}
ARM_LABEL = {
    "oracle": "oracle (published law)",
    "physics": "physics (law, fitted)",
    "pinn": "PINN (law + network)",
    "sr": "symbolic regression",
    "nn": "black-box NN",
}
ARM_ORDER = ["oracle", "physics", "pinn", "sr", "nn"]


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "figure.constrained_layout.use": True,
        }
    )


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, loc="left", fontweight="normal")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    return ax


def save(fig, track: str, name: str):
    p = get_settings().figures(track) / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------


def fig_overview(
    prob,
    fits: dict,
    idx_train=None,
    x_dense=None,
    logx=False,
    logy=False,
    title=None,
    extrap_from=None,
):
    """The data, and what each arm does with it."""
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = prob.x[:, 0]
    if x_dense is None:
        x_dense = (
            np.geomspace(x.min(), x.max(), 400)
            if logx
            else np.linspace(x.min(), x.max(), 400)
        )
    if extrap_from is not None:
        ax.axvspan(extrap_from, x_dense.max(), color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.text(
            extrap_from,
            ax.get_ylim()[1],
            "  extrapolation",
            va="top",
            ha="left",
            color=INK_MUTED,
            fontsize=9,
        )
    for arm in ARM_ORDER:
        if arm not in fits:
            continue
        try:
            yq = fits[arm].predict(x_dense.reshape(-1, 1))
        except Exception:
            continue
        ax.plot(
            x_dense,
            yq,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
            ls="--" if arm == "oracle" else "-",
            lw=1.8 if arm == "oracle" else 2.0,
            zorder=2,
        )
    ax.plot(
        x,
        prob.y,
        "o",
        mfc="none",
        mec=INK,
        mew=1.4,
        ls="none",
        label="measured",
        zorder=3,
    )
    if idx_train is not None:
        ax.plot(
            x[idx_train],
            prob.y[idx_train],
            "o",
            color=INK,
            ms=5,
            ls="none",
            label="training points",
            zorder=4,
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style(ax, title, prob.xlabel, prob.ylabel)
    ax.legend(loc="best", fontsize=9, labelcolor=INK_2)
    return fig


def _agg(df, xcol, ycol, arms=ARM_ORDER):
    g = df.groupby(["arm", xcol])[ycol].median().reset_index()
    return {a: g[g.arm == a].sort_values(xcol) for a in arms if (g.arm == a).any()}


def _declutter(ax, points, min_gap_px=11.0):
    """Push end-of-line labels apart so they never overlap.

    Two arms that converge to the same value put their direct labels on top of
    each other, which is exactly what happened to `pinn` and `oracle` on
    track A. Positions are resolved in display space, where "overlap" actually
    means something, then converted back.
    """
    if not points:
        return []
    order = sorted(range(len(points)), key=lambda i: points[i][1])
    ys = [ax.transData.transform((points[i][0], points[i][1]))[1] for i in order]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap_px:
            ys[k] = ys[k - 1] + min_gap_px
    out = [None] * len(points)
    inv = ax.transData.inverted()
    for k, i in enumerate(order):
        x_px = ax.transData.transform((points[i][0], points[i][1]))[0]
        out[i] = inv.transform((x_px, ys[k]))[1]
    return out


def fig_curve(
    df, xcol, ycol, title, xlabel, ylabel, logx=False, logy=True, annotate=True
):
    """A sweep: one line per arm, median over the reporting seeds.

    Five series, so a legend is required; it sits below the axes rather than
    on the data. The direct labels at the right are de-collided.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    series = _agg(df, xcol, ycol)
    for arm, d in series.items():
        ax.plot(
            d[xcol],
            d[ycol],
            marker="o",
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
            ls="--" if arm == "oracle" else "-",
            lw=1.8 if arm == "oracle" else 2.0,
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=9,
        labelcolor=INK_2,
    )
    if annotate:
        fig.canvas.draw()
        arms = [
            a for a, d in series.items() if len(d) and np.isfinite(d[ycol].iloc[-1])
        ]
        pts = [(series[a][xcol].iloc[-1], series[a][ycol].iloc[-1]) for a in arms]
        for arm, (x_, _), y_ in zip(arms, pts, _declutter(ax, pts)):
            ax.annotate(
                arm,
                (x_, y_),
                textcoords="offset points",
                xytext=(7, 0),
                color=ARM_COLOR[arm],
                fontsize=9,
                va="center",
                annotation_clip=False,
            )
    return fig


def fig_extrapolation_bars(df, title, ylabel="nRMSE"):
    """In-range against out-of-range error, per arm. Log scale, because the
    interesting numbers span six decades."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    arms = [a for a in ARM_ORDER if (df.arm == a).any()]
    w, xs = 0.38, np.arange(len(arms))
    inn = [df[df.arm == a]["nrmse_in"].median() for a in arms]
    out = [df[df.arm == a]["nrmse_out"].median() for a in arms]
    floor = 1e-12
    inn = [max(v, floor) if np.isfinite(v) else floor for v in inn]
    out = [max(v, floor) if np.isfinite(v) else floor for v in out]
    for i, a in enumerate(arms):
        ax.bar(
            xs[i] - w / 2,
            inn[i],
            w * 0.94,
            color=ARM_COLOR[a],
            alpha=0.42,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.bar(
            xs[i] + w / 2,
            out[i],
            w * 0.94,
            color=ARM_COLOR[a],
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.annotate(
            f"{out[i]:.2g}",
            (xs[i] + w / 2, out[i]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms], fontsize=9)
    ax.set_yscale("log")
    _style(ax, title, None, ylabel)
    ax.bar(np.nan, np.nan, color=INK_MUTED, alpha=0.42, label="inside training range")
    ax.bar(np.nan, np.nan, color=INK_MUTED, label="outside training range")
    ax.legend(fontsize=9, labelcolor=INK_2, loc="best")
    return fig


def fig_physics_weight(df, title, param_key=None, published=None):
    """The dial: what is the physics term in the loss worth?

    Two stacked panels rather than two y-scales on one. Held-out error and
    the error on the recovered constant are different measures; putting them
    on a shared axis would be the dual-axis anti-pattern.
    """
    use_style()
    show_param = bool(param_key and f"param_{param_key}" in df and published)
    fig, axes = plt.subplots(
        2 if show_param else 1,
        1,
        sharex=True,
        figsize=(6.6, 5.6 if show_param else 3.8),
    )
    axes = np.atleast_1d(axes)
    d = df.groupby("w_phys").median(numeric_only=True).reset_index()
    wp = d["w_phys"].to_numpy(dtype=float)
    pos = wp[wp > 0]
    # w_phys = 0 is a real setting (no physics term); place it a decade below
    # the smallest positive weight so it is visible on a log axis, and say so.
    xs = np.where(wp <= 0, (pos.min() / 10.0 if len(pos) else 1e-4), wp)

    axes[0].plot(xs, d["nrmse_out"], marker="o", color=ARM_COLOR["pinn"])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    _style(axes[0], title, None, "held-out nRMSE")

    if show_param:
        err = np.abs(d[f"param_{param_key}"] - published) / abs(published) * 100.0
        axes[1].plot(xs, err, marker="o", color=ARM_COLOR["physics"])
        axes[1].set_yscale("log")
        _style(axes[1], None, None, f"|{param_key} - published| / published  [%]")
    axes[-1].set_xlabel(
        "physics weight $w_{phys}$   (leftmost point is $w=0$)", color=INK_2
    )
    return fig


def fig_bars(
    labels,
    values,
    title,
    ylabel,
    colors=None,
    logy=False,
    reference=None,
    reference_label=None,
    fmt="{:.3g}",
):
    """A plain labelled bar chart for one-off comparisons (PN order, residual
    ladder, band coverage). One series, so no legend -- the title names it."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    xs = np.arange(len(labels))
    cols = colors or [ARM_COLOR["physics"]] * len(labels)
    ax.bar(xs, values, 0.62, color=cols, edgecolor=SURFACE, linewidth=2)
    for x_, v_ in zip(xs, values):
        if np.isfinite(v_):
            ax.annotate(
                fmt.format(v_),
                (x_, v_),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
                color=INK_2,
            )
    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, ls="--", lw=1.6)
        # Anchored inside the axes on the left. Placing it past the last bar
        # put it outside the axes, where it was clipped away.
        ax.annotate(
            reference_label or f"{reference:g}",
            (0.01, reference),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=9,
            color=INK_MUTED,
            ha="left",
            va="bottom",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    if logy:
        ax.set_yscale("log")
    _style(ax, title, None, ylabel)
    return fig


def fig_cross_track_extrapolation(summary_rows, title=None):
    """Small multiples: out-of-range error by arm, one panel per track.

    Small multiples put every pair of series on screen together, so this uses
    the all-pairs-validated subset of the palette (blue / orange / aqua /
    violet) on the pinned light surface.
    """
    import pandas as pd

    use_style()
    df = pd.DataFrame(summary_rows)
    tracks = list(dict.fromkeys(df["track"]))
    fig, axes = plt.subplots(
        1, len(tracks), figsize=(3.1 * len(tracks), 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, tr in zip(axes, tracks):
        d = df[df.track == tr]
        arms = [a for a in ARM_ORDER if a in set(d.arm)]
        vals = [max(float(d[d.arm == a]["nrmse_out"].iloc[0]), 1e-12) for a in arms]
        ax.bar(
            np.arange(len(arms)),
            vals,
            0.66,
            color=[ARM_COLOR[a] for a in arms],
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(list(arms), rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        _style(ax, f"track {tr}", None, None)
    axes[0].set_ylabel("nRMSE outside the training range", color=INK_2)
    if title:
        fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    return fig


def fig_parameter_recovery(rows, title="Physical constants recovered from real data"):
    """Recovered / published, on a log axis centred on 1.

    One dot per quantity, with the published value as the line at 1. This is
    the only figure in the project where every track appears at once, and the
    only one whose y-axis is dimensionless on purpose -- the constants have
    incompatible units and a shared linear axis would be meaningless.
    """
    use_style()
    rows = [
        r
        for r in rows
        if r.get("recovered") is not None and r.get("published") not in (None, 0)
    ]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.8))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        ratio = r["recovered"] / r["published"]
        sig = r.get("sigma")
        err = (sig / abs(r["published"])) if sig else None
        col = ARM_COLOR["physics"] if abs(ratio - 1) < 1e-3 else ARM_COLOR["sr"]
        ax.errorbar(
            ratio,
            y,
            xerr=err,
            fmt="o",
            color=col,
            capsize=3,
            ms=7,
            ecolor=col,
            elinewidth=1.6,
        )
    ax.axvline(1.0, color=INK_MUTED, ls="--", lw=1.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['track']} · {r['quantity']}" for r in rows], fontsize=9)
    ax.set_xscale("log")
    _style(ax, title, "recovered / published   (dashed line = published value)", None)
    ax.grid(axis="y", visible=False)
    return fig


def fig_phase2_improvement(df, title=None):
    """What freezing the balanced loss bought the `pinn` arm, per track.

    Form: a dumbbell, not bars. The measure spans four decades so the axis
    must be logarithmic, and a bar encodes length from a zero baseline that a
    log axis does not have. Two dots joined by a line also put the reader on
    the quantity that matters -- how far each cell moved.

    One hue throughout, because every row is the same arm; the row labels
    carry identity, so there is nothing for a second colour to say.
    `relativity/gw150914` shows as a point rather than a dumbbell, which is
    correct: its arm is the ODE-residual PINN, the option does not
    apply to it, and it reports itself not engaged rather than moving.
    """
    use_style()
    rows = list(df.itertuples())
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 2.0))
    colour = ARM_COLOR["pinn"]

    labels = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        labels.append((y, f"{r.track.split('/')[-1]}  ·  {r.question}"))
        moved = r.factor > 1.05
        ax.plot(
            [r.before, r.after],
            [y, y],
            color=colour,
            lw=2.4,
            alpha=0.45,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            [r.before],
            [y],
            "o",
            markersize=7,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=2.2,
            zorder=3,
        )
        ax.plot(
            [r.after],
            [y],
            "o",
            markersize=9,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=3,
        )
        # The label goes to the RIGHT of the worse end, which is always the
        # free side. Left of the better end it collided with the tick labels
        # on exactly the rows that improved most.
        ax.annotate(
            (
                f"  {r.factor:.0f}× better"
                if r.factor >= 10
                else f"  {r.factor:.1f}× better"
            )
            if moved
            else "  not engaged",
            xy=(max(r.before, r.after), y),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=colour if moved else INK_MUTED,
        )

    ax.set_yticks([y for y, _ in labels], [t for _, t in labels])
    ax.set_xscale("log")
    ax.set_xlabel("held-out nRMSE (log scale) — lower is better")
    ax.set_xlim(right=ax.get_xlim()[1] * 40)
    ax.set_ylim(-1.15, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title(title or "What the frozen default bought the PINN arm")
    ax.plot(
        [],
        [],
        "o",
        markerfacecolor=SURFACE,
        markeredgecolor=INK_MUTED,
        markeredgewidth=2.2,
        markersize=7,
        linestyle="none",
        label="before (w_phys = 1, unswitched)",
    )
    ax.plot(
        [],
        [],
        "o",
        color=INK_MUTED,
        markersize=9,
        linestyle="none",
        label="after (gradient-norm balancing)",
    )
    ax.legend(loc="lower right", ncols=2)
    fig.text(
        0.0,
        -0.02,
        "Decided on the tuning seeds (3/7/19), measured on the reporting "
        "seeds (11/23/42).",
        fontsize=8.5,
        color=INK_2,
    )
    return fig


def fig_inverse_potential(inv, truth=None, title=None):
    """From a handful of numbers to a function: the inverse Schrodinger problem.

    Two panels, because the story is a transformation. Left is everything the
    method was told -- a level diagram, which is how a spectroscopist would
    write it down. Right is what came back, against the truth, with the
    states drawn at their own energies in the way every quantum textbook
    draws them.

    `V_true` is a reference rather than a competing series, so it is muted
    ink and heavier, under the recovered curve.
    """
    use_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1, 2.4]}
    )

    # --- left: the given spectrum, as a level diagram ---------------------
    ax = axes[0]
    for e in inv.energies_target:
        ax.plot(
            [0.12, 0.88],
            [e, e],
            color=ARM_COLOR["physics"],
            lw=2.4,
            solid_capstyle="round",
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    lo0 = float(np.min(inv.energies_target))
    hi0 = float(np.max(inv.energies_target))
    pad = 0.12 * ((hi0 - lo0) or 1.0)
    # Headroom for the note, which otherwise lands on the top level.
    ax.set_ylim(lo0 - pad, hi0 + 4 * pad)
    ax.set_ylabel("energy")
    ax.set_title(f"Given: {len(inv.energies_target)} numbers")
    ax.grid(axis="x", visible=False)
    # At the top: the lowest level sits near the bottom of this panel and a
    # note placed there lands on it.
    ax.annotate(
        "no functional form,\nno name, no hint",
        xy=(0.5, 0.99),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )

    # --- right: the recovered potential, with its states ------------------
    ax = axes[1]
    if truth is not None:
        ax.plot(
            inv.x, truth(inv.x), lw=3.4, color=INK_MUTED, zorder=1, label="true V(x)"
        )
    ax.plot(inv.x, inv.v, color=ARM_COLOR["pinn"], zorder=3, label="recovered V(x)")

    span = float(np.ptp(inv.energies_target)) or 1.0
    for k, e in enumerate(inv.energies_achieved):
        ax.plot(
            inv.x,
            np.full_like(inv.x, e),
            color=INK_MUTED,
            lw=0.7,
            ls=(0, (4, 4)),
            zorder=2,
        )
        psi = inv.psi[k]
        amp = 0.16 * span / (np.max(np.abs(psi)) or 1.0)
        ax.plot(
            inv.x,
            e + amp * psi,
            color=ARM_COLOR["physics"],
            lw=1.4,
            alpha=0.85,
            zorder=4,
        )

    lo, hi = float(np.min(inv.energies_target)), float(np.max(inv.energies_target))
    # Headroom above the highest state so the legend has somewhere to sit
    # that is not on top of the potential.
    ax.set_ylim(lo - 0.45 * span, hi + 1.05 * span)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x) and the states, at their energies")
    ax.set_title(title or "Recovered: a function")
    ax.plot([], [], color=ARM_COLOR["physics"], lw=1.4, label="states psi_n")
    ax.legend(loc="upper center", ncols=3)

    err = ""
    if truth is not None:
        err = f"   |V error| where the states live: {inv.error_against(truth):.3g}"
    fig.text(
        0.0,
        -0.02,
        f"Spectrum reproduced to {inv.spectrum_error * 100:.2f}% "
        f"(checked by diagonalising the recovered V, not by the network).{err}",
        fontsize=8.5,
        color=INK_2,
    )
    return fig


def fig_eigen_convergence(states, exact, title=None):
    """Learned eigenvalues against the closed form, and the overlap that says
    whether each level is a distinct state at all.

    Two panels on one figure because they answer two different questions
    about the same run: *is the energy right* and *is it even a new state*.
    """
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    n = np.arange(1, len(states) + 1)
    err = [abs(s.energy - e) / e * 100 for s, e in zip(states, exact, strict=True)]

    ax = axes[0]
    ax.plot(
        n,
        [s.energy for s in states],
        "o",
        markersize=9,
        color=ARM_COLOR["pinn"],
        markeredgecolor=SURFACE,
        markeredgewidth=1.4,
        label="learned",
        zorder=3,
    )
    ax.plot(n, exact, lw=3, color=INK_MUTED, zorder=1, label="exact")
    ax.set_xticks(n)
    ax.set_xlabel("level n")
    ax.set_ylabel("energy")
    ax.set_title(title or "The learned spectrum")
    ax.legend(loc="upper left")

    ax = axes[1]
    colours = [ARM_COLOR["pinn"] if s.converged else ARM_COLOR["sr"] for s in states]
    ax.bar(
        n,
        [max(s.max_overlap, 1e-4) for s in states],
        color=colours,
        edgecolor=SURFACE,
        linewidth=2,
        width=0.6,
    )
    ax.axhline(0.1, color=INK_2, lw=1.2, ls=(0, (3, 3)))
    ax.annotate(
        "collapse threshold",
        xy=(n[-1] + 0.4, 0.1),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xlabel("level n")
    ax.set_ylabel("overlap with the states below")
    ax.set_title("Is it a new state, or a copy?")
    for xi, e in zip(n, err, strict=True):
        ax.annotate(
            f"{e:.2f}%", xy=(xi, 1.3e-4), ha="center", fontsize=7.5, color=INK_MUTED
        )
    return fig


# ---------------------------------------------------------------------------
# method anatomy: what the two PINN forms actually compute
# ---------------------------------------------------------------------------


def _box(ax, xy, w, h, text, face, edge, fontsize=9, weight="normal", ink=None):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.6,
            zorder=2,
        )
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=ink or INK,
        weight=weight,
        zorder=3,
    )


def _arrow(
    ax, start, end, colour=None, label=None, rad=0.0, dy=0.03, dx=0.0, ha="center"
):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.4,
            color=colour or INK_MUTED,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + dx,
            (start[1] + end[1]) / 2 + dy,
            label,
            ha=ha,
            va="bottom",
            fontsize=7.6,
            color=INK_2,
            zorder=3,
        )


def fig_pinn_anatomy():
    """The two PINN forms in this package, drawn as what they compute.

    They are not variants of one architecture -- they answer different
    questions, and the choice follows from whether the law is a differential
    equation or an algebraic relation. Showing them side by side is the
    fastest way to see that `w_phys` means something different in each.

    A diagram rather than a plot: there is no data here, only structure, and
    the structure is the thing that is usually described in three paragraphs
    of prose and still misunderstood.
    """
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    tint = "#eaf1fb"
    tint_pinn = "#e6f6f0"

    # --- A: residual PINN -------------------------------------------------
    ax = axes[0]
    ax.set_title("A · Residual PINN — the network IS the solution", loc="left")
    _box(ax, (0.02, 0.62), 0.16, 0.14, "t\ncollocation", SURFACE, INK_MUTED, 8.4)
    _box(
        ax, (0.26, 0.62), 0.22, 0.14, "NN$_\\theta$(t)", tint, ARM_COLOR["physics"], 10
    )
    _box(ax, (0.58, 0.62), 0.18, 0.14, "y(t)", SURFACE, INK_MUTED, 10)
    _arrow(ax, (0.18, 0.69), (0.26, 0.69))
    _arrow(ax, (0.48, 0.69), (0.58, 0.69))

    _box(ax, (0.58, 0.40), 0.18, 0.13, "dy/dt", SURFACE, INK_MUTED, 9)
    # beside the arrow, not under it: the box below occludes a centred label
    _arrow(
        ax,
        (0.67, 0.62),
        (0.67, 0.53),
        label="autograd",
        dy=-0.012,
        dx=0.015,
        ha="left",
    )
    _box(
        ax,
        (0.22, 0.17),
        0.54,
        0.16,
        "residual   R = dy/dt − f(y; $\\theta_{phys}$)",
        tint_pinn,
        ARM_COLOR["pinn"],
        9.6,
    )
    _arrow(ax, (0.58, 0.465), (0.50, 0.33), rad=-0.15)
    _arrow(ax, (0.58, 0.62), (0.40, 0.33), rad=0.18)
    ax.text(
        0.49,
        0.08,
        "loss  =  data MSE  +  $w_{phys}\\,\\overline{R^2}$",
        ha="center",
        fontsize=10,
        color=INK,
        weight="bold",
    )
    ax.text(
        0.02,
        0.90,
        "the physical constant $\\theta_{phys}$ lives INSIDE the residual,\n"
        "so it receives a gradient through the physics term",
        fontsize=8.4,
        color=INK_2,
        va="top",
    )
    ax.text(
        0.02,
        0.02,
        "used where the law is a differential equation:  gw150914, T1, T5",
        fontsize=8.2,
        color=INK_MUTED,
    )

    # --- B: law + correction ---------------------------------------------
    ax = axes[1]
    ax.set_title("B · Law + correction — the network is the residue", loc="left")
    _box(ax, (0.02, 0.62), 0.14, 0.14, "x", SURFACE, INK_MUTED, 9)
    _box(
        ax,
        (0.24, 0.72),
        0.30,
        0.14,
        "law(x; $\\theta$)",
        tint_pinn,
        ARM_COLOR["pinn"],
        10,
    )
    _box(
        ax,
        (0.24, 0.50),
        0.30,
        0.14,
        "$\\sigma_y\\cdot$NN(x)",
        tint,
        ARM_COLOR["physics"],
        10,
    )
    _arrow(ax, (0.16, 0.71), (0.24, 0.79), rad=0.12)
    _arrow(ax, (0.16, 0.67), (0.24, 0.57), rad=-0.12)
    _box(ax, (0.64, 0.61), 0.14, 0.14, "y", SURFACE, INK_MUTED, 10)
    _arrow(ax, (0.54, 0.79), (0.64, 0.71), rad=0.12)
    _arrow(ax, (0.54, 0.57), (0.64, 0.65), rad=-0.12)
    ax.text(
        0.49,
        0.36,
        "loss  =  $\\dfrac{\\mathrm{MSE}}{\\sigma_y^2}$  +  "
        "$w_{phys}\\,\\overline{\\mathrm{NN}^2}$",
        ha="center",
        fontsize=10,
        color=INK,
        weight="bold",
    )
    ax.text(
        0.02,
        0.24,
        "$w_{phys}\\to\\infty$   the correction is crushed → the arm IS the classical fit\n"
        "$w_{phys}=0$        the correction is free → a black box in a physics hat",
        fontsize=8.6,
        color=INK_2,
        va="top",
    )
    ax.text(
        0.02,
        0.02,
        "used where the law is an algebraic relation:  kepler, hydrogen, cmb",
        fontsize=8.2,
        color=INK_MUTED,
    )
    return fig


def fig_ablation_effect(summary, title=None):
    """Which switches help, which hurt, and by how much.

    Form: a dot plot on a log ratio axis with 1.0 as the reference. Not bars
    -- a ratio has no zero, and a bar drawn from one would be arithmetic
    nonsense. Left of the line is better, right is worse, and the distance is
    the effect size.

    The verdict rides in the tick label rather than as a right-hand column,
    which is the only placement that cannot collide with a clipped point.
    """
    use_style()
    options = list(dict.fromkeys(summary.option))
    fig, ax = plt.subplots(figsize=(8.6, 0.56 * len(options) + 2.4))
    ax.axvline(1.0, color=INK_2, lw=1.4, zorder=1)

    labels = []
    for i, option in enumerate(options):
        y = len(options) - 1 - i
        sub = summary[summary.option == option]
        helped = int((sub.ratio < 0.9).sum())
        hurt = int((sub.ratio > 1.1).sum())
        ships = bool(helped) and not hurt
        colour = ARM_COLOR["pinn"] if ships else ARM_COLOR["sr"]
        verdict = (
            f"ships · helps {helped}"
            if ships
            else (f"hurts {hurt}" if hurt else "no effect")
        )
        labels.append((y, f"{option}\n{verdict}"))
        ax.scatter(
            sub.ratio.clip(1e-2, 1e2),
            np.full(len(sub), y),
            s=64,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=3,
        )

    ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=8.6)
    ax.set_xscale("log")
    ax.set_xlim(4e-2, 3e2)
    ax.set_ylim(-0.9, len(options) - 0.4)
    ax.set_xlabel("error ratio to the unswitched arm  (log) — left is better")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or "One switch at a time, on the tuning seeds")
    ax.annotate(
        "no effect",
        xy=(1.0, len(options) - 0.45),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.4,
        color=INK_2,
        va="center",
    )
    ax.scatter([], [], s=64, color=ARM_COLOR["pinn"], label="shipped")
    ax.scatter([], [], s=64, color=ARM_COLOR["sr"], label="rejected")
    ax.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.30))
    return fig


def fig_alpha_convergence(rows, published=1.0, title=None):
    """A result that is still moving with step size is not a result.

    One point per (stencil order, step), with the published value as a rule.
    The y axis is the distance from that value, logarithmic, because the
    interesting range spans five decades -- and the eye should read "how far
    from Einstein", not "what number came out".
    """
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    orders = sorted({int(r["fd_order"]) for r in rows})
    palette = {2: ARM_COLOR["nn"], 4: ARM_COLOR["sr"], 6: ARM_COLOR["pinn"]}
    for order in orders:
        pts = [r for r in rows if int(r["fd_order"]) == order]
        steps = [float(str(r["step"]).rstrip("m")) for r in pts]
        dev = [abs(float(r["alpha_GR"]) - published) for r in pts]
        idx = np.argsort(steps)
        ax.plot(
            np.array(steps)[idx],
            np.array(dev)[idx],
            "-o",
            color=palette.get(order, INK_MUTED),
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.3,
            label=f"{order}{ {2: 'nd', 4: 'th', 6: 'th'}.get(order, 'th') } order stencil".replace(
                " ", ""
            ),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Plain minutes: a reader should not have to decode "2 x 10^2" back into
    # the step size they chose.
    steps_all = sorted({float(str(r["step"]).rstrip("m")) for r in rows})
    ax.set_xticks(steps_all, [f"{int(v)}" for v in steps_all])
    ax.minorticks_off()
    ax.set_xlabel("finite-difference step (minutes, log scale)")
    ax.set_ylabel("|alpha − 1|   (log)")
    ax.set_title(title or "The 56-sigma refutation was truncation error")
    ax.legend(loc="lower right")
    ax.annotate(
        "a 13% violation of GR\nat 56 formal sigma",
        xy=(180, 0.134),
        xytext=(18, -34),
        textcoords="offset points",
        fontsize=8.4,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    return fig


# ---------------------------------------------------------------------------
# the neglected-terms study
# ---------------------------------------------------------------------------

_NEGLECT_COLOUR = {
    "physics": ARM_COLOR["physics"],
    "pinn": ARM_COLOR["pinn"],
    "nn": ARM_COLOR["nn"],
}


def fig_neglected_sweep(df, key, xlabel, title=None, metric="nrmse_in", logx=False):
    """Three arms against one dial, medians over the reporting seeds.

    The question is always where the lines CROSS, so the arms are direct
    labelled at the right edge and there is no legend box competing for the
    same space.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    g = df.groupby([key, "arm"])[metric].median().unstack("arm")
    for arm in ("physics", "pinn", "nn"):
        if arm not in g:
            continue
        ax.plot(
            g.index,
            g[arm],
            "-o",
            color=_NEGLECT_COLOUR[arm],
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.3,
            label=arm,
        )
        ax.annotate(
            f"  {arm}",
            xy=(g.index[-1], g[arm].iloc[-1]),
            color=_NEGLECT_COLOUR[arm],
            fontsize=9,
            va="center",
        )
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("held-out nRMSE (log) — lower is better")
    ax.set_title(title or "Where the physics prior pays")
    ax.set_xlim(right=g.index[-1] + 0.14 * (g.index[-1] - g.index[0]))
    ax.legend(loc="upper left")
    return fig


def fig_learned_correction(system, correction_fn, r=None, title=None):
    """Did the network learn the physics that was left out of the law?

    The single most informative plot in the study: the correction the PINN
    learned, drawn against the term it was never shown. If they agree, the
    network has recovered missing physics rather than absorbing noise.
    """
    use_style()
    r = np.linspace(system.r_min, system.r_max, 400) if r is None else r
    truth = system.neglected(r)
    learned = correction_fn(r)

    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    ax = axes[0]
    ax.plot(
        r,
        truth,
        lw=3.2,
        color=INK_MUTED,
        zorder=1,
        label="the term left out of the law",
    )
    ax.plot(
        r, learned, color=ARM_COLOR["pinn"], zorder=3, label="what the network learned"
    )
    ax.axhline(0, color=GRID, lw=1.0, zorder=0)
    ax.set_xlabel("r")
    ax.set_ylabel("contribution to y")
    ax.set_title(title or "The network recovers the missing physics")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(r, system.law(r), lw=2.4, color=ARM_COLOR["physics"], label="the law")
    ax.plot(r, truth, lw=2.4, color=INK_MUTED, label="what it misses")
    ax.set_yscale("log")
    # A Gaussian tail underflows to 1e-21 and drags the axis with it, which
    # makes the comparison that matters -- law against missing term -- invisible.
    peak = float(max(np.max(np.abs(truth)), 1e-12))
    ax.set_ylim(peak * 1e-3, float(np.max(system.law(r))) * 2)
    ax.set_xlabel("r")
    ax.set_ylabel("magnitude (log)")
    ax.set_title("for scale")
    ax.legend(loc="upper right")
    fig.text(
        0.0,
        -0.03,
        f"missing term is {system.neglected_fraction * 100:.1f}% of the law "
        f"on average; shape = {system.shape!r}",
        fontsize=8.5,
        color=INK_2,
    )
    return fig


def fig_learning_curves(history, published=None, title=None):
    """What the loss actually did, split into its terms.

    A single total loss hides the trade the physics weight is making. The two
    terms are drawn apart, and the trainable constant beside them, because
    "the loss went down" and "the constant converged" are different claims
    and only the second one is physics.
    """
    use_style()
    has_param = "GM" in history and len(history.get("GM", []))
    fig, axes = plt.subplots(
        1,
        2 if has_param else 1,
        figsize=(10.0 if has_param else 6.0, 3.8),
        squeeze=False,
    )
    ax = axes[0][0]
    ax.semilogy(
        history["epoch"],
        history["data"],
        color=ARM_COLOR["physics"],
        label="data term   MSE / sd_y^2",
    )
    if "phys" in history:
        ax.semilogy(
            history["epoch"],
            history["phys"],
            color=ARM_COLOR["pinn"],
            label="physics term   mean(NN^2)",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term (log)")
    ax.set_title(title or "The loss, split into what it is trading")
    ax.legend(loc="upper right")

    if has_param:
        ax = axes[0][1]
        ax.plot(history["epoch"], history["GM"], color=ARM_COLOR["pinn"])
        if published is not None:
            ax.axhline(published, color=INK_2, lw=1.4)
            ax.annotate(
                "published value",
                xy=(history["epoch"][0], published),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8.4,
                color=INK_2,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("recovered constant")
        ax.set_title("The constant, converging separately")
    return fig


def fig_learned_force(system, force_fn, predict=None, physics=None, title=None):
    """The differential-equation analogue: did the network learn the missing
    FORCE, not merely a curve that happens to fit?

    The left panel plots the correction against **whatever the missing force
    actually depends on** -- angle for the anharmonic term, velocity for
    damping. Plotting a velocity-dependent force against angle draws a flat
    line and says nothing, which is exactly the mistake this parameter
    exists to prevent.

    The right panel is the consequence of having dropped it, against the
    FITTED harmonic arm rather than the harmonic law at the true omega: the
    `physics` arm is free to bias omega, and drawing it unbiased would
    overstate how badly it does.
    """
    use_style()
    t, exact = system.solve(600)
    dexact = np.gradient(exact, t)
    velocity_dependent = system.shape == "damping"

    if velocity_dependent:
        q = np.linspace(dexact.min(), dexact.max(), 300)
        truth = system.missing_force(np.zeros_like(q), q)
        learned = force_fn(np.zeros_like(q), q)
        xlabel, qplot = "angular velocity (rad/s)", q
        left_title = title or "A force that depends on VELOCITY"
        caption_extra = f"gamma = {system.gamma}"
    else:
        amp = abs(system.amplitude)
        q = np.linspace(-amp, amp, 300)
        truth = system.missing_force(q)
        learned = force_fn(q)
        xlabel, qplot = "angle (degrees)", np.degrees(q)
        left_title = title or "A force that depends on ANGLE"
        caption_extra = (
            f"true period {system.true_period:.3f} against the "
            f"modelled {2 * np.pi / system.omega:.3f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    ax.axhline(0, color=GRID, lw=1.0, zorder=0)
    ax.axvline(0, color=GRID, lw=1.0, zorder=0)
    ax.plot(qplot, truth, lw=3.2, color=INK_MUTED, zorder=1, label="the force left out")
    ax.plot(
        qplot,
        learned,
        color=ARM_COLOR["pinn"],
        zorder=3,
        label="learned by the network",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("force left out of the model")
    ax.set_title(left_title)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(t, exact, lw=3.0, color=INK_MUTED, zorder=1, label="the truth")
    harmonic = (
        physics(t)
        if physics is not None
        else system.amplitude * np.cos(system.omega * t)
    )
    ax.plot(
        t,
        harmonic,
        color=ARM_COLOR["physics"],
        lw=1.8,
        ls=(0, (4, 3)),
        zorder=2,
        label="harmonic, fitted" if physics is not None else "harmonic law",
    )
    if predict is not None:
        ax.plot(t, predict(t), color=ARM_COLOR["pinn"], lw=1.6, zorder=3, label="PINN")
    ax.set_xlabel("time")
    ax.set_ylabel("angle (rad)")
    ax.set_title("and what the model cannot reproduce")
    ax.legend(loc="lower left", ncols=2)
    fig.text(
        0.0,
        -0.03,
        f"amplitude {np.degrees(abs(system.amplitude)):.0f} deg, shape "
        f"{system.shape!r} -- the dropped force is "
        f"{system.missing_fraction * 100:.1f}% of the restoring force; "
        f"{caption_extra}",
        fontsize=8.5,
        color=INK_2,
    )
    return fig


def _sequential_cmap():
    """One hue, light to dark. Magnitude has no poles, so it gets one hue."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "physprior_seq", ["#f4f8fd", "#a8c8ec", "#2a78d6", "#123a6b"]
    )


def _diverging_cmap():
    """Two hues with a NEUTRAL midpoint -- never a hue at zero, or the eye
    reads a feature where the data has none."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "physprior_div",
        ["#123a6b", "#2a78d6", "#a8c8ec", "#efeeea", "#f6bfa4", "#eb6834", "#8c3413"],
    )


def fig_pde_field(system, alpha_fitted, title=None):
    """What the modelled equation can and cannot reproduce, as fields.

    Three panels: the truth, the modelled law at its best-fitting constant,
    and the difference. The difference panel is where the answer is: a SMALL
    residual means the missing term was largely absorbable and it is the
    constant that went wrong, while a large structured one means the model
    cannot represent the physics at any constant. Both residuals have
    structure -- the distinction is magnitude, and the diffusive case's is
    six times smaller.
    """
    use_style()
    t, x, truth = system.solve(60)

    modelled = type(system)(
        eps=0.0,
        noise=0.0,
        shape=system.shape,
        alpha=alpha_fitted,
        n_x=system.n_x,
        t_max=system.t_max,
        length=system.length,
    )
    _, _, model_field = modelled.solve(60)
    diff = truth - model_field

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    extent = [x[0], x[-1], t[0], t[-1]]
    vmax = float(np.max(np.abs(truth)))
    for ax, field, name, cmap, lim in (
        (axes[0], truth, "the truth", _sequential_cmap(), (0, vmax)),
        (
            axes[1],
            model_field,
            f"pure diffusion at its best alpha = {alpha_fitted:.4f}",
            _sequential_cmap(),
            (0, vmax),
        ),
        (
            axes[2],
            diff,
            "what the model cannot reproduce",
            _diverging_cmap(),
            (
                -float(np.max(np.abs(diff))) or -1e-12,
                float(np.max(np.abs(diff))) or 1e-12,
            ),
        ),
    ):
        im = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lim[0],
            vmax=lim[1],
        )
        ax.set_xlabel("x")
        ax.set_title(name, fontsize=10)
        ax.grid(visible=False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("t")
    fig.suptitle(title or f"missing term: {system.shape!r}", x=0.0, ha="left")
    fig.text(
        0.0,
        -0.06,
        f"peak residual {np.max(np.abs(diff)):.2e} against a signal of "
        f"{vmax:.2f} -- "
        + (
            "small: the missing term was mostly absorbed into alpha, so it is "
            "the CONSTANT that went wrong"
            if np.max(np.abs(diff)) < 0.02 * vmax
            else "large and structured: no value of alpha can reproduce this"
        ),
        fontsize=8.5,
        color=INK_2,
    )
    return fig


def fig_derivative_accuracy(
    x,
    u_exact,
    u_net,
    uxx_exact,
    uxx_net,
    alpha_curve,
    alpha_true,
    title=None,
):
    """Why a small data loss does not buy a right constant.

    The single most useful picture in this repository for understanding what
    a PINN actually optimises. The left panel shows the network's field
    against the truth and they lie on top of each other -- the data loss is
    8e-4. The middle panel shows the SECOND derivative of the same two
    curves, and they are not close: the network carries high-frequency
    content that is invisible in the value and enormous in the curvature.

    Since the constant being identified here is exactly the ratio
    |u_t| / |u_xx|, that excess curvature is the error, and no amount of
    loss balancing removes it -- it is not a weighting problem. The right
    panel is the fix: penalise the wiggle one derivative ABOVE the one the
    equation reads, and the recovered constant walks back to the truth.

    `alpha_curve` is `(weights, alphas)`; weights are drawn as categories,
    not on a log axis, because the sweep includes zero.
    """
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    exact_kw = dict(color=INK_MUTED, lw=2.2, ls="--", zorder=2)
    net_kw = dict(color=ARM_COLOR["pinn"], lw=2.0, zorder=3)

    axes[0].plot(x, u_exact, label="exact solution", **exact_kw)
    axes[0].plot(x, u_net, label="network", **net_kw)
    _style(axes[0], "The field: indistinguishable", "x", "u(x, t)")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].axhline(0, color=GRID, lw=1.2, zorder=1)
    axes[1].plot(x, uxx_exact, label="exact solution", **exact_kw)
    axes[1].plot(x, uxx_net, label="network", **net_kw)
    _style(
        axes[1],
        "Its second derivative: not close",
        "x",
        r"$\partial^2 u/\partial x^2$",
    )
    axes[1].legend(frameon=False, loc="upper right")

    weights, alphas = alpha_curve
    pos = np.arange(len(weights))
    axes[2].axhline(
        alpha_true, color=INK_MUTED, lw=2.0, ls="--", zorder=2, label="true value"
    )
    axes[2].plot(pos, alphas, "o-", ms=8, **net_kw)
    for xp, a in zip(pos, alphas, strict=False):
        axes[2].annotate(
            f"{a:.3f}",
            (xp, a),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=INK_2,
            fontsize=9,
        )
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels(["0" if w == 0 else f"{w:g}" for w in weights], color=INK_2)
    axes[2].set_ylim(0, max(alpha_true, max(alphas)) * 1.45)
    _style(
        axes[2],
        "Penalise the wiggle and the constant returns",
        "weight on the curvature penalty",
        r"recovered $\alpha$",
    )
    axes[2].legend(frameon=False, loc="lower right")

    if title:
        fig.suptitle(title, color=INK, x=0.005, ha="left", fontsize=13)
    return fig
