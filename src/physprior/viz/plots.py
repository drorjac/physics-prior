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
    the honest picture: its arm is the ODE-residual PINN, the option does not
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
