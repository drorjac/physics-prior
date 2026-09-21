"""GIF writers, so the simulations and the training can be watched rather
than summarised.

Everything goes through matplotlib's PillowWriter: ffmpeg is not installed on
this machine, and a GIF renders in a notebook, in a README and on GitHub
without a codec. Frame counts are kept small (about 120) so the files stay in
the low hundreds of kilobytes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from physprior.viz.plots import (
    ARM_COLOR,
    GRID,
    INK,
    INK_2,
    INK_MUTED,
    SURFACE,
    _style,
    use_style,
)

# The categorical slots validated all-pairs on the light surface, reused for
# bodies in an orbit plot. Past four bodies, colour stops carrying identity
# and the trail shape does the work instead.
BODY_COLORS = [
    "#eda100",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#4a3aa7",
    "#e87ba4",
    "#008300",
    "#e34948",
]


# GIF frames are full-colour bitmaps, so file size scales with figure area x
# dpi^2 x frames. 100 dpi keeps a 7-inch figure readable while holding a
# 120-frame animation to a few hundred kB.
GIF_DPI = 100


def _save(anim, path: Path, fps: int, dpi: int = GIF_DPI) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close("all")
    return path


def animate_orbits(
    traj,
    path,
    *,
    frames: int = 120,
    fps: int = 20,
    trail: int = 10_000,
    scale: float = 1.0,
    unit: str = "AU",
    title: str = "",
    equal: bool = True,
    sizes: list[float] | None = None,
) -> Path:
    """Bodies moving, with trails. `scale` divides the positions."""
    use_style()
    r = np.asarray(traj.r) / scale
    n_t, n_b = r.shape[0], r.shape[1]
    idx = np.linspace(0, n_t - 1, min(frames, n_t)).astype(int)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lim = float(np.abs(r[..., :2]).max()) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if equal:
        ax.set_aspect("equal")
    _style(ax, title, f"x  [{unit}]", f"y  [{unit}]")

    names = traj.names or [f"body {i}" for i in range(n_b)]
    sizes = sizes or [11 if i == 0 else 7 for i in range(n_b)]
    trails = [
        ax.plot([], [], lw=1.4, alpha=0.75, color=BODY_COLORS[i % len(BODY_COLORS)])[0]
        for i in range(n_b)
    ]
    dots = [
        ax.plot(
            [],
            [],
            "o",
            ms=sizes[i],
            color=BODY_COLORS[i % len(BODY_COLORS)],
            mec=SURFACE,
            mew=1.5,
            label=names[i],
        )[0]
        for i in range(n_b)
    ]
    clock = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", color=INK_2, fontsize=9
    )
    if n_b <= 6:
        ax.legend(loc="lower right", fontsize=8, labelcolor=INK_2)

    t_unit, t_div = _time_unit(traj.t)

    def update(k):
        j = idx[k]
        lo = max(0, j - trail)
        for i in range(n_b):
            trails[i].set_data(r[lo : j + 1, i, 0], r[lo : j + 1, i, 1])
            dots[i].set_data([r[j, i, 0]], [r[j, i, 1]])
        clock.set_text(f"t = {traj.t[j] / t_div:,.1f} {t_unit}")
        return [*trails, *dots, clock]

    return _save(
        FuncAnimation(fig, update, frames=len(idx), blit=True), Path(path), fps
    )


def _time_unit(t):
    span = float(t[-1] - t[0])
    if span > 3.2e7 * 2:
        return "years", 3.15576e7
    if span > 86400 * 3:
        return "days", 86400.0
    if span > 3600 * 3:
        return "hours", 3600.0
    return "s", 1.0


def animate_training(
    history,
    x,
    y,
    predict_at,
    path,
    *,
    frames: int = 60,
    fps: int = 12,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    truth=None,
    logx=False,
    logy=False,
    param_name: str | None = None,
    param_true: float | None = None,
    x_grid: np.ndarray | None = None,
) -> Path:
    """Watch a model learn.

    Left panel: the fit at each recorded epoch against the data. Right panel:
    the loss, and -- the part that matters for this project -- the physical
    parameter walking toward (or away from) its published value.
    """
    use_style()
    epochs = np.asarray(history["epoch"])
    idx = np.linspace(0, len(epochs) - 1, min(frames, len(epochs))).astype(int)
    # When the caller replays recorded snapshots, the grid MUST be the one
    # those snapshots were taken on, or the curve is drawn against the wrong
    # abscissa. Passing it explicitly is the only safe option.
    if x_grid is not None:
        xq = np.asarray(x_grid, float)
    else:
        xq = (
            np.geomspace(np.min(x), np.max(x), 300)
            if logx
            else np.linspace(np.min(x), np.max(x), 300)
        )

    two = param_name is not None and param_name in history
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(11.2 if two else 6.2, 4.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.plot(x, y, "o", mfc="none", mec=INK, mew=1.3, ls="none", label="data")
    if truth is not None:
        ax.plot(xq, truth(xq), "--", color=INK_MUTED, lw=1.6, label="true law")
    (line,) = ax.plot([], [], color=ARM_COLOR["pinn"], lw=2.2, label="model")
    tag = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", color=INK_2, fontsize=9
    )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(xq.min(), xq.max())
    pad = 0.12 * (np.max(y) - np.min(y) or 1.0)
    ax.set_ylim(np.min(y) - pad, np.max(y) + pad)
    _style(ax, title, xlabel, ylabel)
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK_2)

    if two:
        ax2 = axes[1]
        pv = np.asarray(history[param_name], float)
        if param_true:
            ax2.axhline(param_true, color=INK_MUTED, ls="--", lw=1.6)
            ax2.annotate(
                f" published {param_true:.6g}",
                (0.02, param_true),
                xycoords=("axes fraction", "data"),
                fontsize=9,
                color=INK_MUTED,
                va="bottom",
            )
        (pline,) = ax2.plot([], [], color=ARM_COLOR["physics"], lw=2.2)
        ax2.set_xlim(epochs[0], epochs[-1])
        lo, hi = float(np.min(pv)), float(np.max(pv))
        m = 0.08 * (hi - lo or abs(hi) or 1.0)
        ax2.set_ylim(lo - m, hi + m)
        _style(ax2, f"the trainable constant: {param_name}", "epoch", param_name)

    def update(k):
        j = idx[k]
        line.set_data(xq, predict_at(j, xq))
        tag.set_text(f"epoch {epochs[j]:,}")
        out = [line, tag]
        if two:
            pline.set_data(epochs[: j + 1], pv[: j + 1])
            out.append(pline)
        return out

    return _save(
        FuncAnimation(fig, update, frames=len(idx), blit=False), Path(path), fps
    )


def animate_wavefunction(
    x,
    psi_t,
    t,
    path,
    *,
    potential=None,
    frames: int = 120,
    fps: int = 20,
    title: str = "",
    xlabel: str = "x",
    energy: float | None = None,
) -> Path:
    """|psi|^2 evolving, with the potential drawn behind it."""
    use_style()
    idx = np.linspace(0, len(t) - 1, min(frames, len(t))).astype(int)
    dens = np.abs(psi_t) ** 2
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if potential is not None:
        v = np.asarray(potential, float)
        finite = np.isfinite(v)
        vs = v.copy()
        if finite.any():
            top = np.percentile(v[finite], 98)
            vs = np.clip(v, None, top)
            vs = (vs - vs.min()) / (np.ptp(vs) or 1.0) * dens.max() * 0.9
        ax.fill_between(x, 0, vs, color=GRID, lw=0, zorder=0, label="potential V(x)")
    (line,) = ax.plot([], [], color=ARM_COLOR["pinn"], lw=2.2, label=r"$|\psi|^2$")
    tag = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", color=INK_2, fontsize=9
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, dens.max() * 1.15)
    _style(ax, title, xlabel, r"$|\psi(x,t)|^2$")
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK_2)

    def update(k):
        j = idx[k]
        line.set_data(x, dens[j])
        extra = f"   E = {energy:.4g}" if energy is not None else ""
        tag.set_text(f"t = {t[j]:.3g}{extra}")
        return [line, tag]

    return _save(
        FuncAnimation(fig, update, frames=len(idx), blit=True), Path(path), fps
    )
