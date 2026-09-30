"""Figures for the Lorenz study, all drawn from results/lorenz.

Colour reuses the project's validated categorical slots (viz.plots): the PINN
takes the PINN slot, the black box the NN slot, multiple shooting the
`physics` slot (it is the classical fit of the law), and the PINN polished by
shooting the one remaining slot. Secondary arms are told apart by line style
and marker, never by a fifth hue.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from physprior.config import get_settings
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

from . import TRACK
from . import study as ST
from .system import LAMBDA1, THETA_NAMES, THETA_TRUE, butterfly_ensemble, reference

COLOR = {
    "pinn": ARM_COLOR["pinn"],
    "nn": ARM_COLOR["nn"],
    "nn_regress": ARM_COLOR["nn"],
    "ms": ARM_COLOR["physics"],
    "shooting": ARM_COLOR["physics"],
    "pinn_polish": ARM_COLOR["sr"],
    "fd_regress": INK_MUTED,
}
LABEL = {
    "pinn": "PINN",
    "nn": "black-box NN",
    "nn_regress": "NN, then regression",
    "ms": "multiple shooting",
    "shooting": "single shooting",
    "pinn_polish": "PINN, then shooting",
    "fd_regress": "finite differences",
}
STYLE = {
    "pinn": {"ls": "-", "marker": "o"},
    "nn": {"ls": "-", "marker": "s"},
    "nn_regress": {"ls": "--", "marker": "s", "mfc": SURFACE},
    "ms": {"ls": "-", "marker": "D"},
    "shooting": {"ls": ":", "marker": "D", "mfc": SURFACE},
    "pinn_polish": {"ls": "-", "marker": "^"},
    "fd_regress": {"ls": ":", "marker": "x"},
}
COMP = ("x", "y", "z")


def _save(fig, name: str):
    p = get_settings().figures(TRACK) / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _ex(key: str) -> dict:
    return ST.load_examples()[key]


# ---------------------------------------------------------------------------
# the method


def fig_block_diagram(save: bool = True):
    """The PINN for Lorenz as a block diagram: what is computed, what is
    trained, and where each loss term comes from."""
    use_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 50)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, color=INK_2, fill=SURFACE, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                fc=fill,
                ec=color,
                lw=1.6,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold" if bold else "normal",
            linespacing=1.4,
        )

    def arrow(p, q, color=INK_2, text=None, dy=1.2):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.3,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )
        if text:
            ax.text(
                (p[0] + q[0]) / 2,
                (p[1] + q[1]) / 2 + dy,
                text,
                ha="center",
                fontsize=8,
                color=INK_2,
            )

    pinn, nn, phys = ARM_COLOR["pinn"], ARM_COLOR["nn"], ARM_COLOR["physics"]
    box(1, 20, 11, 10, "time t\n$s = 2t/T - 1$")
    box(
        16, 20, 14, 10, "Fourier features\n$s,\\ \\sin(k\\pi s/2),$\n$\\cos(k\\pi s/2)$"
    )
    box(34, 17, 16, 16, "MLP, tanh\n4 x 64\n\nweights W\n(trained)", color=nn)
    box(54, 20, 13, 10, "$u(t) = \\mu + sd\\cdot N$\n(x, y, z)")
    arrow((12.5, 25), (15.5, 25))
    arrow((30.5, 25), (33.5, 25))
    arrow((50.5, 25), (53.5, 25))

    # data branch
    box(72, 36, 16, 10, "$L_{data}$\nmean $((u(t_i)-y_i)/sd)^2$", color=nn)
    box(54, 38, 13, 7, "40 noisy\nobservations $y_i$", fill=GRID)
    arrow((67.5, 41.5), (71.5, 41.5))
    arrow((63, 30.5), (75, 35.5))

    # physics branch
    box(54, 3, 13, 11, "$du/dt$\nforward mode,\nsame pass")
    box(72, 3, 16, 11, "$L_{phys}$\nmean $(du/dt - f(u;\\theta))^2$", color=pinn)
    box(
        72,
        19,
        16,
        11,
        "Lorenz law $f$\n$\\sigma(y-x)$\n$x(\\rho-z)-y$,  $xy-\\beta z$",
        color=phys,
    )
    box(
        91,
        19,
        17,
        11,
        "constants $\\theta$\n$\\sigma, \\rho, \\beta$\n(trained, log scale)",
        color=phys,
    )
    arrow((42, 16.5), (57, 14.5))
    ax.text(
        44, 9, "dN/ds carried\nlayer by layer", ha="center", fontsize=8, color=INK_2
    )
    arrow((67.5, 8.5), (71.5, 8.5))
    arrow((80, 18.5), (80, 14.5))
    arrow((90.5, 24.5), (88.5, 24.5))
    arrow((67.5, 24), (71.5, 24))

    # total loss and optimiser
    box(
        92,
        36,
        16,
        10,
        "$L = L_{data} + w(k)\\,L_{phys}$\nAdam (cosine), then\nL-BFGS",
        bold=False,
    )
    arrow((88.5, 41), (91.5, 41))
    # route L_phys around the constants box: right, then up
    ax.plot([88.5, 111, 111], [8.5, 8.5, 41], color=INK_2, lw=1.3)
    arrow((111, 41), (108.6, 41))
    ax.text(
        100,
        1.5,
        "$w(k)$: 0 during warm-up, then ramped to $w_{phys}$",
        ha="center",
        fontsize=8,
        color=INK_2,
    )
    ax.text(
        1,
        47,
        "Physics-informed network for Lorenz-63: trajectory and constants fitted together",
        fontsize=11,
        color=INK,
    )
    return _save(fig, "block_diagram") if save else fig


# ---------------------------------------------------------------------------
# chaos


def fig_butterfly_3d(save: bool = True):
    """Two trajectories that start 1e-8 apart, on the attractor."""
    use_style()
    u0 = np.array(reference(np.array([1.0, 1.0, 20.0]), np.array([0.0, 20.0]))[-1])
    t = np.linspace(0, 30, 6001)
    a = reference(u0, t)
    b = reference(u0 + np.array([1e-8, 0, 0]), t)
    d = np.linalg.norm(a - b, axis=1)
    split = int(np.argmax(d > 1.0))
    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*a[: split + 1].T, color=INK_MUTED, lw=0.7, label="together")
    ax.plot(*a[split:].T, color=ARM_COLOR["physics"], lw=0.8, label="trajectory A")
    ax.plot(*b[split:].T, color=ARM_COLOR["sr"], lw=0.8, label="trajectory B")
    ax.scatter(*u0, color=INK, s=18)
    ax.set_xlabel("x", color=INK_2)
    ax.set_ylabel("y", color=INK_2)
    ax.set_zlabel("z", color=INK_2)
    ax.set_title(
        f"start 1e-8 apart; indistinguishable until t = {t[split]:.0f}",
        color=INK,
        loc="left",
        fontsize=10,
    )
    ax.view_init(elev=18, azim=-60)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)
        pane.set_edgecolor(GRID)
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, a[:, 0], color=ARM_COLOR["physics"], lw=1.2, label="A")
    ax2.plot(t, b[:, 0], color=ARM_COLOR["sr"], lw=1.2, ls="--", label="B")
    ax2.axvline(t[split], color=INK_MUTED, lw=1, ls=":")
    _style(ax2, "x(t) of both", "time", "x")
    ax2.legend(fontsize=8)
    return _save(fig, "butterfly_3d") if save else fig


def fig_separation(save: bool = True):
    """Distance between neighbours grows like exp(lambda t) until it reaches
    the size of the attractor."""
    use_style()
    sep = ST.load("separation")
    meta = _meta()
    lam = meta["chaos"]["lambda_fit"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.semilogy(sep.t, sep.dist, color=INK_MUTED, lw=1, label="one pair")
    ax.semilogy(
        sep.t,
        np.exp(sep.mean_log_dist),
        color=ARM_COLOR["physics"],
        lw=2,
        label=f"geometric mean over {meta['chaos']['n_pairs']} pairs",
    )
    tt = np.linspace(0, 20, 50)
    ax.semilogy(
        tt,
        1e-8 * np.exp(LAMBDA1 * tt),
        color=INK,
        ls="--",
        lw=1,
        label=f"$10^{{-8}} e^{{\\lambda_1 t}}$, $\\lambda_1$ = {LAMBDA1}",
    )
    ax.text(
        21,
        3e-6,
        f"fitted slope {lam:.3f}",
        color=INK_2,
        fontsize=9,
    )
    ax.set_ylim(1e-9, 1e2)
    _style(ax, "The butterfly effect, measured", "time", "|A(t) - B(t)|")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "separation") if save else fig


def gif_butterfly(frames: int = 90):
    """24 trajectories from a ball of radius 1e-5 spreading over the attractor."""
    use_style()
    u0 = np.array(reference(np.array([1.0, 1.0, 20.0]), np.array([0.0, 20.0]))[-1])
    t, U = butterfly_ensemble(u0, n=24, delta=1e-5, t_end=24.0)
    step = max(1, len(t) // frames)
    idx = np.arange(0, len(t), step)
    fig = plt.figure(figsize=(5.0, 4.2), dpi=80)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-22, 22)
    ax.set_ylim(-28, 28)
    ax.set_zlim(2, 50)
    ax.set_xlabel("x", color=INK_2)
    ax.set_ylabel("y", color=INK_2)
    ax.set_zlabel("z", color=INK_2)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)
        pane.set_edgecolor(GRID)
    ref = U[:, 0]
    ax.plot(*ref[::2].T, color=GRID, lw=0.5)
    pts = ax.scatter(*U[0].T, color=ARM_COLOR["sr"], s=14)
    head = ax.scatter(*U[0, 0], color=INK, s=16)
    title = ax.set_title("", color=INK, loc="left", fontsize=10)

    def update(i):
        k = idx[i]
        pts._offsets3d = tuple(U[k].T)
        head._offsets3d = tuple(U[k, :1].T)
        spread = float(np.max(np.linalg.norm(U[k] - U[k].mean(0), axis=1)))
        title.set_text(
            f"t = {t[k]:4.1f}  ({t[k] * LAMBDA1:4.1f} Lyapunov times)   "
            f"spread {spread:.1e}"
        )
        ax.view_init(elev=20, azim=-60)
        return pts, head, title

    # Frames through one shared 64-colour palette: only the moving points
    # change between frames, so Pillow stores each frame as a small delta.
    from PIL import Image

    images = []
    for i in range(len(idx)):
        update(i)
        fig.canvas.draw()
        rgba = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())
        images.append(Image.fromarray(rgba[..., :3]))
    plt.close(fig)
    # palette from a strip of early, middle and late frames
    picks = [images[0], images[len(images) // 2], images[-1]]
    strip = Image.new("RGB", (picks[0].width * 3, picks[0].height))
    for k, im in enumerate(picks):
        strip.paste(im, (k * im.width, 0))
    palette = strip.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames_q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    p = get_settings().figures(TRACK) / "butterfly.gif"
    frames_q[0].save(
        p,
        save_all=True,
        append_images=frames_q[1:],
        duration=66,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return p


def fig_horizon(save: bool = True):
    """How long a forecast stays valid, against how wrong the constants are,
    started from the exact state; the measured arms are overlaid."""
    use_style()
    h = ST.load("horizon")
    med = h.groupby("eps").vpt.median()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.semilogx(
        med.index,
        med.values,
        color=INK_MUTED,
        lw=2,
        label="exact start, constants off by eps",
    )
    ax.fill_between(
        med.index,
        h.groupby("eps").vpt.min(),
        h.groupby("eps").vpt.max(),
        color=GRID,
        alpha=0.8,
        lw=0,
    )
    main = ST.load("main")
    for arm in ("pinn", "pinn_polish", "ms", "nn_regress"):
        d = main[main.arm == arm]
        ax.scatter(
            d.err_theta,
            d.vpt,
            color=COLOR[arm],
            label=LABEL[arm],
            marker=STYLE[arm]["marker"],
            zorder=3,
            edgecolor=SURFACE,
            linewidth=1,
        )
    _style(
        ax,
        "Forecast horizon against error in the constants",
        "mean relative error in (sigma, rho, beta)",
        "valid time (Lyapunov times)",
    )
    ax.legend(fontsize=8)
    return _save(fig, "horizon") if save else fig


# ---------------------------------------------------------------------------
# one fit, seen from inside


def fig_reconstruction(save: bool = True):
    use_style()
    nn, pinn = (
        _ex(f"main/nn/chosen/{ST.EXAMPLE_SEED}"),
        _ex(f"main/pinn/chosen/{ST.EXAMPLE_SEED}"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(pinn["t_dense"], pinn["u_dense"][:, k], color=INK, lw=1, label="truth")
        ax.plot(
            nn["t_dense"],
            nn["pred"][:, k],
            color=COLOR["nn"],
            lw=1.6,
            ls="--",
            label=LABEL["nn"],
        )
        ax.plot(
            pinn["t_dense"],
            pinn["pred"][:, k],
            color=COLOR["pinn"],
            lw=1.6,
            label=LABEL["pinn"],
        )
        ax.scatter(
            pinn["t_obs"],
            pinn["y_obs"][:, k],
            s=16,
            color=INK_2,
            zorder=3,
            label="observations (5 % noise)",
        )
        _style(ax, None, None, COMP[k])
    axes[0].set_title(
        "Reconstruction between 40 noisy observations (seed 11)", loc="left", color=INK
    )
    axes[-1].set_xlabel("time", color=INK_2)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    return _save(fig, "reconstruction") if save else fig


def fig_phase_space(save: bool = True):
    use_style()
    nn, pinn = (
        _ex(f"main/nn/chosen/{ST.EXAMPLE_SEED}"),
        _ex(f"main/pinn/chosen/{ST.EXAMPLE_SEED}"),
    )
    fig = plt.figure(figsize=(11, 4.8))
    for i, (ex, arm) in enumerate(((nn, "nn"), (pinn, "pinn"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(*ex["u_dense"].T, color=INK_MUTED, lw=1, label="truth")
        ax.plot(*ex["pred"].T, color=COLOR[arm], lw=1.5, label=LABEL[arm])
        ax.scatter(*ex["y_obs"].T, color=INK, s=8, label="observations")
        ax.set_title(LABEL[arm], color=INK, loc="left")
        ax.view_init(elev=18, azim=-60)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(SURFACE)
            pane.set_edgecolor(GRID)
        ax.legend(fontsize=8, loc="upper left")
    return _save(fig, "phase_space") if save else fig


def fig_loss_breakdown(
    key: str = f"main/pinn/chosen/{ST.EXAMPLE_SEED}",
    name: str = "loss_breakdown",
    save: bool = True,
):
    """Every term of the loss, the weight on the physics term, the learning
    rate, and the three constants, over training."""
    use_style()
    h = _ex(key)["history"]
    step = np.asarray(h["step"], float)
    lb = np.asarray(h["phase"]) == "lbfgs"
    t_lb = step[lb].min() if lb.any() else None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.semilogy(step, h["data"], color=COLOR["nn"], label="$L_{data}$")
    ax.semilogy(step, h["phys"], color=COLOR["pinn"], label="$L_{phys}$ (all)")
    for comp, ls in zip(COMP, (":", "--", "-."), strict=True):
        ax.semilogy(
            step,
            h[f"phys_{comp}"],
            color=COLOR["pinn"],
            lw=1,
            ls=ls,
            label=f"$L_{{phys}}$, d{comp}/dt",
        )
    ax.axhline(0.05**2, color=INK_MUTED, lw=1, ls="--")
    ax.text(
        step.max() * 0.02,
        0.05**2 * 1.4,
        "noise floor of $L_{data}$",
        fontsize=8,
        color=INK_2,
    )
    _style(ax, "Loss terms", "step", "loss")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[0, 1]
    ax.plot(step, h["w"], color=COLOR["pinn"], label="physics weight w")
    _style(ax, "Curriculum: the physics weight", "step", "w")
    ax = axes[1, 1]
    ax.semilogy(step[~lb], np.asarray(h["lr"])[~lb], color=INK_2)
    _style(ax, "Adam learning rate", "step", "learning rate")
    ax = axes[1, 0]
    for name_, c, ls in zip(THETA_NAMES, THETA_TRUE, ("-", "--", ":"), strict=True):
        ax.plot(
            step,
            np.asarray(h[name_]) / c,
            color=COLOR["pinn"],
            ls=ls,
            label=f"{name_} / true",
        )
    ax.axhline(1.0, color=INK, lw=1)
    _style(ax, "Constants, relative to the truth", "step", "ratio")
    ax.legend(fontsize=8)
    for a in axes.flat:
        if t_lb is not None:
            a.axvline(t_lb, color=INK_MUTED, lw=1, ls=":")
    if t_lb is not None:
        axes[0, 0].text(
            t_lb, axes[0, 0].get_ylim()[1], " L-BFGS", fontsize=8, color=INK_2, va="top"
        )
    return _save(fig, name) if save else fig


def fig_forecast(save: bool = True):
    use_style()
    nn, pinn = (
        _ex(f"main/nn/chosen/{ST.EXAMPLE_SEED}"),
        _ex(f"main/pinn/chosen/{ST.EXAMPLE_SEED}"),
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for ex, arm in ((nn, "nn"), (pinn, "pinn")):
        ax.semilogy(
            ex["forecast_t"] * LAMBDA1,
            ex["forecast_err"],
            color=COLOR[arm],
            label=LABEL[arm]
            + (" (extrapolated)" if arm == "nn" else " (law with fitted constants)"),
        )
    ax.axhline(0.4, color=INK_MUTED, ls="--", lw=1)
    ax.text(0.1, 0.45, "validity threshold 0.4", fontsize=8, color=INK_2)
    _style(
        ax,
        "Forecast past the end of the window (seed 11)",
        "Lyapunov times after the window",
        "standardised error",
    )
    ax.legend(fontsize=8)
    return _save(fig, "forecast") if save else fig


# ---------------------------------------------------------------------------
# sweeps


def _sweep(ax, df, x, y, arms, logx=True):
    for arm in arms:
        d = df[df.arm == arm]
        if d.empty or d[y].isna().all():
            continue
        g = d.groupby(x)[y]
        med = g.median()
        st = STYLE[arm]
        ax.plot(
            med.index,
            med.values,
            color=COLOR[arm],
            ls=st["ls"],
            marker=st["marker"],
            mfc=st.get("mfc", COLOR[arm]),
            label=LABEL[arm],
        )
        ax.scatter(d[x], d[y], color=COLOR[arm], s=10, alpha=0.45, lw=0)
    ax.set_yscale("log")
    if logx and x == "noise":
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(left=0)
    elif logx:
        ax.set_xscale("log")


SWEEPS = {
    "noise": ("noise", "noise sd (fraction of each component's sd)"),
    "budget": ("n_obs", "number of observations"),
}


def fig_sweep(stage: str, save: bool = True):
    use_style()
    x, xl = SWEEPS[stage]
    df = ST.load(stage)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    _sweep(axes[0], df, x, "state", ("nn", "pinn", "ms", "pinn_polish"))
    _style(axes[0], "Trajectory error", xl, "normalised RMSE")
    _sweep(
        axes[1],
        df,
        x,
        "err_theta",
        ("nn_regress", "pinn", "ms", "pinn_polish", "fd_regress"),
    )
    _style(axes[1], "Error in (sigma, rho, beta)", xl, "mean relative error")
    _sweep(axes[2], df, x, "deriv", ("nn", "pinn", "ms", "pinn_polish"))
    _style(axes[2], "Derivative error", xl, "normalised RMSE")
    axes[0].legend(fontsize=8)
    # the empty corner differs: low noise leaves the lower left full
    axes[1].legend(fontsize=8, loc="lower right" if stage == "noise" else "lower left")
    return _save(fig, stage) if save else fig


def fig_ladder(save: bool = True):
    use_style()
    df = ST.load("ladder")
    order = list(ST.LADDER)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, col, title in zip(
        axes,
        ("state", "err_theta", "seconds"),
        ("Trajectory error", "Error in the constants", "Training time, s"),
        strict=True,
    ):
        for i, name in enumerate(order):
            d = df[df.label == name][col]
            final = name == ST.LADDER_FINAL
            c = COLOR["pinn"] if final else INK_2
            ax.scatter(d, [i] * len(d), color=c, s=14, alpha=0.5, lw=0)
            ax.scatter([d.median()], [i], color=c, s=60, marker="o", zorder=3)
        if col != "seconds":
            ax.set_xscale("log")
        _style(ax, title)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    return _save(fig, "ladder") if save else fig


def fig_speed(save: bool = True):
    use_style()
    sp = ST.load("speed")
    lab = [
        f"{d}{' + torch.compile' if c else ''}"
        for d, c in zip(sp.derivative, sp.compiled, strict=True)
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(sp))
    ax.barh(
        y,
        sp.ms_per_step,
        color=[
            COLOR["pinn"] if (d == "forward" and not c) else INK_MUTED
            for d, c in zip(sp.derivative, sp.compiled, strict=True)
        ],
        height=0.6,
    )
    for yi, v in zip(y, sp.ms_per_step, strict=True):
        if np.isfinite(v):
            ax.text(v, float(yi), f" {v:.1f} ms", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    _style(
        ax,
        "One training step, by how du/dt is computed (CPU, 1 thread)",
        "milliseconds per Adam step",
    )
    return _save(fig, "speed") if save else fig


def fig_main(save: bool = True):
    use_style()
    df = ST.load("main")
    arms = ["nn", "nn_regress", "fd_regress", "shooting", "ms", "pinn", "pinn_polish"]
    cols = (
        ("state", "Trajectory error"),
        ("err_theta", "Error in the constants"),
        ("vpt", "Valid forecast (Lyapunov times)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (col, title) in zip(axes, cols, strict=True):
        for i, arm in enumerate(arms):
            d = df[df.arm == arm][col].dropna()
            if d.empty:
                continue
            ax.scatter(d, [i] * len(d), color=COLOR[arm], s=14, alpha=0.5, lw=0)
            ax.scatter(
                [d.median()],
                [i],
                color=COLOR[arm],
                s=60,
                marker=STYLE[arm]["marker"],
                zorder=3,
            )
        if col != "vpt":
            ax.set_xscale("log")
        _style(ax, title)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([LABEL[a] for a in arms])
    axes[0].invert_yaxis()
    return _save(fig, "main") if save else fig


def _meta() -> dict:
    from physprior.io import load_json

    return load_json(TRACK, "meta")


def worst_vanilla_seed() -> int:
    """The reporting seed on which the vanilla PINN does worst."""
    lad = ST.load("ladder")
    v = lad[lad.label == "vanilla"]
    return int(v.loc[v.state.idxmax(), "seed"])


def regenerate(gif: bool = True) -> list:
    out = [
        fig_block_diagram(),
        fig_butterfly_3d(),
        fig_separation(),
        fig_horizon(),
        fig_reconstruction(),
        fig_phase_space(),
        fig_loss_breakdown(),
        fig_loss_breakdown(
            f"ladder/pinn/vanilla/{worst_vanilla_seed()}", "loss_breakdown_vanilla"
        ),
        fig_forecast(),
        fig_sweep("noise"),
        fig_sweep("budget"),
        fig_ladder(),
        fig_speed(),
        fig_main(),
    ]
    if gif:
        out.append(gif_butterfly())
    return out
