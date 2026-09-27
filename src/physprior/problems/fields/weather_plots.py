"""Figures for fields/weather. Read from results/, never from a live fit,
except the station-level panels, which refit the deterministic arms.

Colour: the four competing arms keep the project's validated slots. The two
kriging baselines add magenta (universal kriging) and ochre (ordinary
kriging); the six together pass the palette validator on the light surface
over all pairs (worst CVD dE 6.1, above the 6.0 floor; worst normal-vision
dE 16.3), and they also differ by marker, so colour is never the only cue.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physprior.benchmark.protocol import fit_arm
from physprior.io import load_json, load_table
from physprior.methods.base import REPORT_SEEDS
from physprior.viz import plots as P

from . import weather as W
from . import weather_sim as S

FIG = "fields"
COLOR = P.ARM_COLOR | {
    "uk": "#c93f8e",
    "gp": "#8a6d00",
    "pinn_static": P.ARM_COLOR["pinn"],
}
LABEL = {
    "oracle": "oracle (Gamma = -6.5)",
    "physics": "physics",
    "pinn": "PINN",
    "pinn_static": "PINN, fixed start",
    "sr": "SR",
    "nn": "NN",
    "gp": "ordinary kriging",
    "uk": "universal kriging",
}
MARKER = {
    "oracle": "_",
    "physics": "o",
    "pinn": "s",
    "pinn_static": "x",
    "sr": "D",
    "nn": "^",
    "gp": "v",
    "uk": "P",
}
ORDER = ["oracle", "physics", "pinn", "pinn_static", "sr", "nn", "gp", "uk"]
CASE_TITLE = {"july": "July 2023, 12 UTC", "january": "January 2023, 12 UTC"}


def _adjusted(x, y, theta):
    """T moved to the box centre with the fitted horizontal gradients, so a
    temperature-height plot shows the height dependence alone."""
    return y - theta["a"] * (x[:, 1] - W.LON0) - theta["b"] * (x[:, 2] - W.LAT0)


def _map(ax, lon, lat, c, cmap, vmin, vmax, **kw):
    sc = ax.scatter(
        lon, lat, c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=kw.pop("s", 22), **kw
    )
    ax.set_aspect(1.0 / np.cos(np.radians(W.LAT0)))
    ax.set_xlabel("longitude  [deg E]", color=P.INK_2)
    ax.set_ylabel("latitude  [deg N]", color=P.INK_2)
    return sc


# ---------------------------------------------------------------------------


def fig_data() -> None:
    """Stations on the map, and temperature against height for both months."""
    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    st = load_table(W.track("july"), "stations")
    sc = _map(
        axes[0],
        st.lon,
        st.lat,
        st.elev_m,
        P._sequential_cmap(),
        0,
        float(st.elev_m.max()),
        edgecolor=P.INK_MUTED,
        linewidths=0.3,
    )
    test = st.split == "test"
    axes[0].scatter(
        st.lon[test], st.lat[test], s=60, facecolor="none", edgecolor=P.INK, lw=0.9
    )
    fig.colorbar(sc, ax=axes[0], label="station height  [m]", shrink=0.8)
    P._style(axes[0], "ISD stations; circled = highest 25% (test)")
    for ax, case in zip(axes[1:], ("july", "january"), strict=True):
        prob, _ = W.problem(case)
        head = load_json(W.track(case), "meta")["headline"]
        th = head["physics"]["params"]
        z = prob.x[:, 0]
        ya = _adjusted(prob.x, prob.y, th)
        ax.scatter(z, ya, s=14, color=P.INK_2, alpha=0.7, label="stations")
        zz = np.linspace(0, z.max(), 50)
        ax.plot(
            zz,
            th["T0"] + th["Gamma"] * zz,
            color=COLOR["physics"],
            label=f"law fit, Gamma = {th['Gamma']:.2f} K/km",
        )
        zbar, ybar = float(np.mean(z)), float(np.mean(ya))
        ax.plot(
            zz,
            ybar + W.GAMMA_STD * (zz - zbar),
            color=P.INK_MUTED,
            ls="--",
            label="-6.5 K/km through the mean",
        )
        P._style(
            ax,
            CASE_TITLE[case],
            "station height z  [km]",
            "T at box centre  [degC]",
        )
        ax.legend(fontsize=8.5, labelcolor=P.INK_2)
    P.save(fig, FIG, "weather_data")


def _fits_on_split(prob, itr, seed, arms):
    fits = {a: fit_arm(a, prob, itr, seed, sr_fast=True) for a in arms}
    for k in ("gp", "uk"):
        fits[k] = W.fit_kriging(prob, itr, k)
    return fits


def fig_extrapolation(case: str, arms=("physics", "pinn", "nn")) -> None:
    """Up the mountain: what each arm predicts for the highest stations."""
    P.use_style()
    prob, _ = W.problem(case)
    itr, ite = W.elevation_split(prob)
    fits = _fits_on_split(prob, itr, REPORT_SEEDS[0], arms)
    th = load_json(W.track(case), "meta")["headline"]["physics"]["params"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ya = _adjusted(prob.x, prob.y, th)
    ax.scatter(prob.x[itr, 0], ya[itr], s=12, color=P.GRID, label="train stations")
    ax.scatter(prob.x[ite, 0], ya[ite], s=16, color=P.INK, label="test stations")
    for name, f in fits.items():
        p = _adjusted(prob.x[ite], np.asarray(f.predict(prob.x[ite])), th)
        ax.scatter(
            prob.x[ite, 0],
            p,
            s=18,
            marker=MARKER[name],
            color=COLOR[name],
            alpha=0.85,
            label=LABEL[name],
        )
    ax.axvline(prob.x[itr, 0].max(), color=P.INK_MUTED, ls=":", lw=1)
    P._style(
        ax,
        f"{CASE_TITLE[case]}: trained below the dotted line",
        "station height z  [km]",
        "T at box centre  [degC]",
    )
    ax.legend(fontsize=8, labelcolor=P.INK_2, ncol=2)
    ax = axes[1]
    lim = [float(prob.y[ite].min()) - 1, float(prob.y[ite].max()) + 1]
    ax.plot(lim, lim, color=P.INK_MUTED, ls="--", lw=1)
    for name, f in fits.items():
        ax.scatter(
            prob.y[ite],
            f.predict(prob.x[ite]),
            s=16,
            marker=MARKER[name],
            color=COLOR[name],
            alpha=0.85,
            label=LABEL[name],
        )
    P._style(
        ax,
        f"test stations, seed {REPORT_SEEDS[0]}",
        "measured T  [degC]",
        "predicted T  [degC]",
    )
    P.save(fig, FIG, f"weather_extrapolation_{case}")


def fig_maps(case: str, arms=("physics", "nn")) -> None:
    """Measured field; residual of the law fitted to every station; and each
    arm's error at the held-out mountain stations."""
    P.use_style()
    prob, _ = W.problem(case)
    itr, ite = W.elevation_split(prob)
    fits = _fits_on_split(prob, itr, REPORT_SEEDS[0], arms)
    full = fit_arm("physics", prob, np.arange(len(prob)), REPORT_SEEDS[0])
    lon, lat = prob.x[:, 1], prob.x[:, 2]
    panels = ["data", "residual", *fits]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.0, 4.0 * nrow))
    axes = axes.ravel()
    sc = _map(
        axes[0],
        lon,
        lat,
        prob.y,
        "RdYlBu_r",
        float(prob.y.min()),
        float(prob.y.max()),
    )
    fig.colorbar(sc, ax=axes[0], label="T  [degC]", shrink=0.8)
    P._style(axes[0], f"{CASE_TITLE[case]}: measured")
    res = prob.y - full.predict(prob.x)
    v = float(np.max(np.abs(res)))
    sc = _map(axes[1], lon, lat, res, P._diverging_cmap(), -v, v)
    fig.colorbar(sc, ax=axes[1], label="data - law  [K]", shrink=0.8)
    P._style(axes[1], "residual of the law fitted to all stations")
    errs = {n: f.predict(prob.x[ite]) - prob.y[ite] for n, f in fits.items()}
    v = float(max(np.max(np.abs(e)) for e in errs.values()))
    for ax, (name, e) in zip(axes[2:], errs.items(), strict=False):
        ax.scatter(lon[itr], lat[itr], s=6, color=P.GRID)
        sc = _map(ax, lon[ite], lat[ite], e, P._diverging_cmap(), -v, v, s=30)
        rm = float(np.sqrt(np.mean(e**2)))
        P._style(ax, f"{LABEL[name]}: test error, RMSE {rm:.2f} K")
        # one shared scale, so every panel's colour means the same kelvin
        fig.colorbar(sc, ax=ax, label="predicted - measured  [K]", shrink=0.8)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    P.save(fig, FIG, f"weather_maps_{case}")


def _bar_panel(ax, df, title, arms):
    arms = [a for a in arms if (df.arm == a).any()]
    xs = np.arange(len(arms))
    med = [float(df[df.arm == a].nrmse_out.median()) for a in arms]
    for i, a in enumerate(arms):
        # the fixed-start PINN is a diagnostic: hatched, so it does not read
        # as a second PINN arm
        diag = a == "pinn_static"
        ax.bar(
            xs[i],
            med[i],
            0.7,
            color=P.SURFACE if diag else COLOR[a],
            edgecolor=COLOR[a] if diag else P.SURFACE,
            hatch="//" if diag else None,
            linewidth=1.5 if diag else 2,
        )
        pts = df[df.arm == a].nrmse_out.to_numpy()
        ax.scatter(np.full(len(pts), xs[i]), pts, s=8, color=P.INK, zorder=3)
        ax.annotate(
            f"{med[i]:.2g}",
            (xs[i], med[i]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=P.INK_2,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[a] for a in arms], rotation=30, ha="right", fontsize=8.5)
    P._style(ax, title, None, "held-out nRMSE")


def _combined(tr: str, kriging: bool = True) -> pd.DataFrame:
    ex = load_table(tr, "extrapolation")
    if not kriging:
        return ex
    kr = load_table(tr, "kriging")
    return pd.concat([ex, kr[kr.sweep == "elevation"]], ignore_index=True)


def fig_nrmse() -> None:
    P.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharey="row")
    for j, case in enumerate(("july", "january")):
        tr = W.track(case)
        _bar_panel(axes[0, j], _combined(tr), f"{CASE_TITLE[case]}: elevation", ORDER)
        try:
            bl = load_table(tr, "blocks")
            kr = load_table(tr, "kriging")
            # one number per (arm, seed): the pooled RMS over the four blocks
            df = pd.concat([bl, kr[kr.sweep == "block"]], ignore_index=True)
            df["seed"] = df["seed"].fillna(-1)
            df = (
                df.assign(sq=df.nrmse_out**2)
                .groupby(["arm", "seed"])
                .sq.mean()
                .pipe(np.sqrt)
                .rename("nrmse_out")
                .reset_index()
            )
            _bar_panel(axes[1, j], df, f"{CASE_TITLE[case]}: longitude blocks", ORDER)
        except FileNotFoundError:
            axes[1, j].set_visible(False)
    P.save(fig, FIG, "weather_nrmse")


def fig_gamma() -> None:
    """The lapse rate each arm implies (mean dT/dz over the stations), fitted
    on the elevation training split, against the reference."""
    P.use_style()
    panels = [
        (W.track("july"), "July: real", W.GAMMA_STD, "-6.5 K/km (ICAO)"),
        (W.track("january"), "January: real", W.GAMMA_STD, "-6.5 K/km (ICAO)"),
    ]
    sm = load_json(S.TRACK_ROOT, "meta")
    for case in S.CASES:
        panels.append(
            (
                S.track(case),
                f"simulated: {case}",
                sm["gamma_best_linear"][case],
                "best linear Gamma of the true field",
            )
        )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (tr, title, ref, reflabel) in zip(axes, panels, strict=True):
        try:
            df = _combined(tr, kriging=not tr.startswith(S.TRACK_ROOT))
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        arms = [a for a in ORDER if (df.arm == a).any()]
        for i, a in enumerate(arms):
            v = df[df.arm == a].dTdz.to_numpy()
            ax.scatter(
                np.full(len(v), i), v, color=COLOR[a], marker=MARKER[a], s=40, zorder=3
            )
        ax.axhline(ref, color=P.INK_MUTED, ls="--", lw=1.2, label=reflabel)
        if tr.startswith(S.TRACK_ROOT):
            ax.axhline(
                S.TRUTH["Gamma"], color=P.INK_2, ls=":", lw=1.2, label="free-air truth"
            )
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(
            [LABEL[a] for a in arms], rotation=35, ha="right", fontsize=8
        )
        P._style(ax, title, None, "mean dT/dz  [K/km]")
        ax.legend(fontsize=8, labelcolor=P.INK_2)
    P.save(fig, FIG, "weather_gamma")


def fig_coverage() -> None:
    """Pulls (Gamma - ref) / sigma over simulated draws, physics vs uk."""
    P.use_style()
    cov = load_table(S.TRACK_ROOT, "coverage")
    fig, axes = plt.subplots(1, len(S.CASES), figsize=(11.0, 4.0), sharey=True)
    bins = np.linspace(-8, 8, 33)
    for ax, case in zip(axes, S.CASES, strict=True):
        for arm in ("physics", "uk"):
            z = cov[(cov.case == case) & (cov.arm == arm)].z.clip(-8, 8)
            ax.hist(
                z,
                bins=bins,
                histtype="step",
                lw=2,
                color=COLOR[arm],
                label=f"{LABEL[arm]} (sd {z.std():.2f})",
            )
        P._style(
            ax,
            f"simulated: {case}",
            "(Gamma - reference) / quoted sigma",
            "draws",
        )
        ax.legend(fontsize=8.5, labelcolor=P.INK_2)
    P.save(fig, FIG, "weather_sim_coverage")


def all_figures(cases=("july", "january")) -> None:
    fig_data()
    for case in cases:
        fig_extrapolation(case)
        fig_maps(case)
    fig_nrmse()
    fig_gamma()
    fig_coverage()
