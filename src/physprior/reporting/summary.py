"""The project summary notebook, and the plots it draws.

    physprior summary [--execute]

One notebook for a reader who wants the results fast. Every section shows the
data, then the result, then a one-line conclusion whose numbers are computed
from the committed `results/` files when the notebook runs. Nothing here
retrains a model; the only computation is a pendulum integration and one
closed-form field, each well under a second of work.

`show(task)` draws one task's plots and prints its conclusion, so a reader can
pick a task by name from `TASKS`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from physprior.config import get_settings
from physprior.io import load_json, load_table
from physprior.reporting.cells import _nb, code, md
from physprior.viz import plots as P

SHORT = {
    "oracle": "oracle",
    "physics": "physics",
    "pinn": "PINN",
    "sr": "SR",
    "nn": "NN",
}
COMPETITORS = ["physics", "pinn", "sr", "nn"]
# ICAO Doc 7488/3 (1993) / ISO 2533:1975 standard-atmosphere lapse rate, K/km
LAPSE_STANDARD = -6.5
LOWER = "lower is better"


def _results(*parts: str):
    return get_settings().results_dir.joinpath(*parts)


def _med(df: pd.DataFrame, col: str, by: str = "arm") -> pd.Series:
    return df.groupby(by)[col].median()


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "inf"
    return f"{v:.2g}" if abs(v) < 0.1 else f"{v:.2f}"


def _arm_bars(ax, df: pd.DataFrame, col: str, arms, colors=None, labels=None):
    """Median over seeds as a large marker, each seed as a small dot; the oracle
    as a reference line. Markers rather than bars, because the axes are log and
    a bar's length on a log axis means nothing."""
    colors = colors or {}
    labels = labels or {}
    for i, arm in enumerate(arms):
        v = df.loc[df.arm == arm, col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        c = colors.get(arm, P.ARM_COLOR.get(arm))
        ax.plot(
            [i - 0.25, i + 0.25],
            [np.median(v)] * 2,
            color=c,
            lw=3.5,
            solid_capstyle="round",
        )
        ax.plot(
            np.full(len(v), i), v, "o", color=c, mec=P.SURFACE, mew=1.2, ms=8, ls="none"
        )
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([labels.get(a, SHORT.get(a, a)) for a in arms])
    if (df.arm == "oracle").any():
        o = float(df.loc[df.arm == "oracle", col].median())
        ax.axhline(o, color=P.ARM_COLOR["oracle"], ls="--", lw=1.6)
        ax.text(-0.55, o, "oracle", va="bottom", ha="left", color=P.INK_2, fontsize=9)
    ax.grid(axis="x", visible=False)


def _pinn_wins(df: pd.DataFrame, other: str) -> tuple[int, int]:
    w = df.pivot_table(index="seed", columns="arm", values="nrmse_out")
    return int((w["pinn"] < w[other]).sum()), len(w)


# ---------------------------------------------------------------------------
# a. relativity/gw150914
# ---------------------------------------------------------------------------

GW = "relativity/gw150914"


def gw150914_data():
    meta = load_json(GW, "meta")
    ex = load_table(GW, "extrapolation")
    t, f = np.asarray(meta["t_ms"]), np.asarray(meta["f_hz"])
    k = int(ex.n_train.iloc[0])
    order = np.argsort(t)
    tr, te = order[:k], order[k:]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(t[te].min() - 2, t.max() + 3, color=P.GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(t[tr], f[tr], "o", color=P.INK, label="training cycles")
    ax.plot(t[te], f[te], "o", mfc="none", mec=P.INK, mew=1.5, label="held out (later)")
    ax.text(
        t[te].min(), f.max(), " extrapolation", va="top", color=P.INK_MUTED, fontsize=9
    )
    P._style(
        ax,
        "GW150914: frequency track of the chirp (LIGO H1+L1)",
        "time from merger reference [ms]",
        "gravitational-wave frequency [Hz]",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=P.INK_2)
    return fig


def gw150914_result():
    ex = load_table(GW, "extrapolation")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _arm_bars(ax, ex, "nrmse_out", COMPETITORS)
    ax.set_yscale("log")
    P._style(
        ax,
        "Error on the held-out later cycles",
        None,
        f"nRMSE out of range (log, {LOWER})",
    )
    return fig


def gw150914_conclusion() -> str:
    ex = load_table(GW, "extrapolation")
    m = _med(ex, "nrmse_out")
    best = m[COMPETITORS].idxmin()
    w, n = _pinn_wins(ex, "physics")
    return (
        f"Best extrapolation: {SHORT[best]}. Median nRMSE out of range: "
        + ", ".join(f"{SHORT[a]} {_fmt(m[a])}" for a in COMPETITORS)
        + f"; the ODE-residual PINN beats the fitted law on {w}/{n} seeds."
    )


# ---------------------------------------------------------------------------
# b. neglected terms
# ---------------------------------------------------------------------------

NEGLECTED_AMP = 60  # degrees, the amplitude docs/neglected/README.md tabulates


def neglected_data():
    from physprior.benchmark.neglected import NeglectedODE

    sim = NeglectedODE(amplitude=np.radians(NEGLECTED_AMP), noise=0.02, shape="damping")
    t, theta = sim.solve()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(t, np.degrees(theta), color=P.INK, label="truth: pendulum with damping")
    ax.plot(
        t,
        NEGLECTED_AMP * np.cos(sim.omega * t),
        color=P.ARM_COLOR["oracle"],
        ls="--",
        lw=1.6,
        label="the law given: undamped, harmonic",
    )
    P._style(
        ax,
        f"Simulated pendulum released at {NEGLECTED_AMP} degrees",
        "time [1/omega]",
        "angle [degrees]",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=9,
        labelcolor=P.INK_2,
    )
    return fig


def neglected_result():
    df = load_table("", "neglected_ode")
    d = df[df["shape"] == "damping"]
    g = d.groupby(["arm", "amplitude_deg"]).nrmse.median().unstack(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm in ("physics", "pinn", "nn"):
        ax.plot(g.index, g[arm], "o-", color=P.ARM_COLOR[arm], label=P.ARM_LABEL[arm])
    ax.set_yscale("log")
    P._style(
        ax,
        "Damping left out of the law: trajectory error",
        "initial amplitude [degrees]",
        f"nRMSE (log, {LOWER})",
    )
    ax.legend(fontsize=9, labelcolor=P.INK_2)
    return fig


def neglected_conclusion() -> str:
    df = load_table("", "neglected_ode")
    g = df.groupby(["shape", "amplitude_deg", "arm"]).nrmse.median().unstack("arm")
    dmp, anh = g.loc["damping"], g.loc["anharmonic"]
    both = int(((dmp.pinn < dmp.physics) & (dmp.pinn < dmp.nn)).sum())
    r = dmp.loc[NEGLECTED_AMP]
    anh_phys = int((anh.physics <= anh[["pinn", "nn"]].min(axis=1)).sum())
    return (
        f"With damping left out, the PINN beats both physics and NN at {both}/{len(dmp)} "
        f"amplitudes (at {NEGLECTED_AMP} deg: PINN {_fmt(r.pinn)}, NN {_fmt(r.nn)}, "
        f"physics {_fmt(r.physics)}); when the missing term is shaped like the law "
        f"(anharmonic), physics is best at {anh_phys}/{len(anh)}."
    )


# ---------------------------------------------------------------------------
# c. pulsars
# ---------------------------------------------------------------------------

PSR = "gravity/pulsar_spindown"
PSR_SIM = "gravity/pulsar_spindown/sim"


def pulsars_data(y_range=(0.0, 5.0)):
    meta = load_json(PSR, "meta")
    ps = pd.DataFrame(meta["pulsars"])
    age = np.log10(ps.age_yr.to_numpy())
    n_obs = ps.n_obs.to_numpy()
    k = int(meta["dipole_test"]["n_train"])
    order = np.argsort(age)
    train = np.zeros(len(age), bool)
    train[order[:k]] = True
    n_fit, sig = meta["dipole_test"]["n_fit"], meta["dipole_test"]["n_sigma"]
    lo, hi = y_range
    # Older pulsars' indices run to +-80 (glitch recovery); drawn on a linear
    # axis they would flatten the question, so they sit on the edge as arrows.
    off = (n_obs < lo) | (n_obs > hi)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(
        3.0, color=P.ARM_COLOR["oracle"], ls="--", lw=1.6, label="dipole law, n = 3"
    )
    ax.axhspan(n_fit - sig, n_fit + sig, color=P.ARM_COLOR["physics"], alpha=0.2, lw=0)
    ax.axhline(
        n_fit,
        color=P.ARM_COLOR["physics"],
        lw=1.6,
        label="n fitted on the younger half, +/- 1 sigma",
    )
    ax.plot(age[train], n_obs[train], "o", color=P.INK, label="younger half (training)")
    ax.plot(
        age[~train & ~off],
        n_obs[~train & ~off],
        "o",
        mfc="none",
        mec=P.INK,
        mew=1.5,
        label="older half (held out)",
    )
    for m, y in (("^", hi), ("v", lo)):
        sel = off & ((n_obs > hi) if m == "^" else (n_obs < lo))
        ax.plot(
            age[sel],
            np.full(sel.sum(), y),
            m,
            mfc="none",
            mec=P.INK,
            mew=1.5,
            ms=8,
            clip_on=False,
            ls="none",
        )
    ax.set_ylim(lo, hi)
    P._style(
        ax,
        "Braking index of young pulsars (ATNF catalogue)",
        "log10 characteristic age [yr]",
        "braking index n",
    )
    ax.text(
        age.max(),
        hi * 0.93,
        f"{off.sum()} older pulsars off scale (n from {n_obs.min():.0f} to "
        f"{n_obs.max():.0f})",
        ha="right",
        va="top",
        color=P.INK_MUTED,
        fontsize=8,
    )
    ax.legend(fontsize=8, labelcolor=P.INK_2, loc="lower left")
    return fig


def pulsars_result():
    ex = load_table(PSR_SIM, "extrapolation")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _arm_bars(ax, ex, "nrmse_out", COMPETITORS)
    ax.set_yscale("log")
    P._style(
        ax,
        "Simulated control (n falls with age): error on the older half",
        None,
        f"nRMSE out of range (log, {LOWER})",
    )
    return fig


def pulsars_conclusion() -> str:
    dt = load_json(PSR, "meta")["dipole_test"]
    ex = load_table(PSR_SIM, "extrapolation")
    m = _med(ex, "nrmse_out")
    w, n = _pinn_wins(ex, "physics")
    best = m[COMPETITORS].idxmin()
    return (
        f"Real pulsars: n = {dt['n_fit']:.2f} +/- {dt['n_sigma']:.2f}, "
        f"{dt['deficit_sigma']:.1f} sigma below 3, so the dipole law is incomplete. "
        f"Control: the PINN learns the age dependence (nRMSE {_fmt(m.pinn)} against "
        f"physics {_fmt(m.physics)}, better on {w}/{n} seeds); best arm {SHORT[best]} "
        f"({_fmt(m[best])})."
    )


# ---------------------------------------------------------------------------
# d. field reconstruction
# ---------------------------------------------------------------------------

REC = "reconstruction"
REC_TARGET = 0.1
BLACK_BOXES = ["gp", "interp", "nn"]


def reconstruction_data(seed: int = 11, n: int = 32):
    from physprior.reconstruction import fields as F
    from physprior.reconstruction.study import NOISE, make_scene

    sc = make_scene(F.draw_field("box", 2, seed))
    x, _, _ = F.sensors(sc.field, n, seed, NOISE, sc.scale)
    m = round(float(np.sqrt(len(sc.X))))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(
        sc.truth.reshape(m, m).T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis"
    )
    ax.plot(x[:, 0], x[:, 1], "o", color="white", mec=P.INK, ms=5, label=f"{n} sensors")
    ax.plot(
        *sc.field.centers.T,
        "X",
        color="white",
        mec=P.INK,
        mew=1,
        ms=11,
        ls="none",
        label="true sources",
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="temperature u [arb. units]")
    P._style(ax, "2-D plate: the field to reconstruct", "x [box side]", "y [box side]")
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        labelcolor=P.INK_2,
    )
    return fig


def _rec_table(case: str) -> pd.DataFrame:
    s = load_table(REC, "samples_needed")
    s = s[(s.metric == "nrmse") & (s.target == REC_TARGET) & (s.case == case)]
    t = s.pivot_table(index="d", columns="method", values="N_star")
    t["black_box"] = t[BLACK_BOXES].min(axis=1)
    t["P"] = s.groupby("d").P.first()
    return t


def reconstruction_result(case: str = "box"):
    t = _rec_table(case)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(
        t.index,
        t.physics,
        "o-",
        color=P.ARM_COLOR["physics"],
        label="physics (Green's function fit)",
    )
    ax.plot(
        t.index,
        t.black_box,
        "o-",
        color=P.ARM_COLOR["nn"],
        label="best black box (GP, RBF interpolation or NN)",
    )
    ax.plot(t.index, t.P, ":", color=P.INK_MUTED, label="unknowns in the law")
    ax.set_yscale("log")
    ax.set_xticks(t.index)
    P._style(
        ax,
        f"Sensors needed to reach nRMSE {REC_TARGET} ({case} case)",
        "dimension",
        f"sensors needed (log, {LOWER})",
    )
    ax.legend(fontsize=9, labelcolor=P.INK_2)
    return fig


def reconstruction_conclusion() -> str:
    parts = []
    for case in ("box", "free"):
        t = _rec_table(case)
        ph = "/".join(f"{v:.0f}" for v in t.physics)
        bb = "/".join(f"{v:.0f}" for v in t.black_box)
        parts.append(f"{case}: physics {ph}, best black box {bb}")
    return (
        f"Sensors for nRMSE {REC_TARGET} in 1-D/2-D/3-D -- "
        + "; ".join(parts)
        + ". The physics fit grows with the number of unknowns, the black boxes "
        "by about a decade per dimension."
    )


# ---------------------------------------------------------------------------
# e. learned dynamics
# ---------------------------------------------------------------------------

DYN = "dynamics"
DYN_ARMS = {
    "hnn_leapfrog": ("Hamiltonian net, leapfrog", P.ARM_COLOR["pinn"]),
    "residual": ("residual u + dt NN(u)", P.ARM_COLOR["sr"]),
    "node": ("neural ODE", P.ARM_COLOR["physics"]),
    "direct": ("direct u' = NN(u)", P.ARM_COLOR["nn"]),
}
PENDULUM_DT, PENDULUM_TRAIN_STEPS = 0.1, 200  # docs/dynamics/README.md, systems table


def dynamics_data():
    z = np.load(_results(DYN, "examples.npz"))
    tr = z["ode__pendulum__hnn_leapfrog__test__truth"][0]
    t = np.arange(len(tr)) * PENDULUM_DT
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    t_train = PENDULUM_TRAIN_STEPS * PENDULUM_DT
    ax.axvspan(t_train, t.max(), color=P.GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(t, tr[:, 0], color=P.INK, lw=1.2)
    top = 1.45 * np.abs(tr[:, 0]).max()
    ax.set_ylim(-1.1 * np.abs(tr[:, 0]).max(), top)
    ax.text(
        t_train,
        top,
        " beyond the training horizon",
        va="top",
        color=P.INK_MUTED,
        fontsize=9,
    )
    P._style(
        ax,
        "Pendulum test trajectory the steppers must reproduce",
        "time [sqrt(l/g)]",
        "angle q [rad]",
    )
    return fig


def dynamics_result_energy():
    c = load_table(DYN, "curves")
    c = c[(c.exp == "main") & (c.system == "pendulum") & (c.split == "test")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for arm, (label, color) in DYN_ARMS.items():
        g = c[c.arm == arm].groupby("t").drift_H.median()
        ax.plot(g.index, g.values, color=color, label=label)
    ax.set_yscale("log")
    P._style(
        ax,
        "Pendulum: energy error |H(t) - H(0)| / (H(0) - H_min)",
        "time [sqrt(l/g)]",
        f"relative energy error (log, {LOWER})",
    )
    ax.legend(
        fontsize=9,
        labelcolor=P.INK_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    return fig


def _valid_steps() -> pd.DataFrame:
    d = load_table(DYN, "ode_main")
    d = d[(d.split == "test") & d.arm.isin(["direct", "residual"])]
    return d.groupby(["system", "arm"]).valid_steps.median().unstack("arm")


def dynamics_result_horizon():
    v = _valid_steps()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(v))
    for j, arm in enumerate(("direct", "residual")):
        label, color = DYN_ARMS[arm]
        ax.bar(x + (j - 0.5) * 0.36, v[arm], width=0.34, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(v.index)
    ax.grid(axis="x", visible=False)
    P._style(
        ax,
        "Steps before the rollout error exceeds 0.1 (test set)",
        None,
        "valid steps (median, higher is better)",
    )
    ax.legend(fontsize=9, labelcolor=P.INK_2)
    return fig


def dynamics_conclusion() -> str:
    d = load_table(DYN, "ode_main")
    p = d[(d.system == "pendulum") & (d.split == "test")].groupby("arm").H_late.median()
    v = _valid_steps()
    ratio = (v.residual / v.direct).median()
    return (
        "Pendulum energy error late in the rollout: leapfrog HNN "
        f"{p.hnn_leapfrog:.1e}, against {min(p.direct, p.node, p.residual):.1e} to "
        f"{max(p.direct, p.node, p.residual):.1e} for the black boxes; the residual "
        f"stepper stays valid a median {ratio:.1f}x longer than the direct one "
        f"({(v.residual > v.direct).sum()}/{len(v)} systems)."
    )


# ---------------------------------------------------------------------------
# f. weather over the Alps
# ---------------------------------------------------------------------------

WX = "fields/weather/july"


def weather_data():
    st = load_table(WX, "stations")
    tr = st.split == "train"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    sc = a1.scatter(st.lon, st.lat, c=st.temp_c, cmap="viridis", s=22)
    fig.colorbar(sc, ax=a1, label="July mean temperature [deg C]")
    a1.set_aspect(1.0 / np.cos(np.radians(st.lat.mean())))
    P._style(a1, f"{len(st)} stations, July 2023", "longitude [deg]", "latitude [deg]")
    a2.plot(
        st.elev_m[tr] / 1e3,
        st.temp_c[tr],
        "o",
        color=P.INK,
        ms=4,
        label="training (low)",
    )
    a2.plot(
        st.elev_m[~tr] / 1e3,
        st.temp_c[~tr],
        "o",
        mfc="none",
        mec=P.INK,
        ms=5,
        label="held out (high)",
    )
    P._style(
        a2,
        "Temperature against elevation",
        "elevation [km]",
        "July mean temperature [deg C]",
    )
    a2.legend(fontsize=9, labelcolor=P.INK_2)
    return fig


def _weather_tables():
    ex = load_table(WX, "extrapolation")
    kr = load_table(WX, "kriging")
    kr = kr[kr.sweep == "elevation"]
    return ex, kr


def weather_result():
    ex, kr = _weather_tables()
    df = pd.concat([ex[ex.arm.isin([*COMPETITORS, "oracle"])], kr], ignore_index=True)
    arms = [*COMPETITORS, "uk", "gp"]
    labels = {"uk": "universal\nkriging", "gp": "GP"}
    colors = {"uk": P.INK_2, "gp": P.INK_MUTED}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    _arm_bars(ax, df, "nrmse_out", arms, colors, labels)
    ax.set_yscale("log")
    P._style(
        ax,
        "Predicting the high stations from the low ones",
        None,
        f"nRMSE on the high stations (log, {LOWER})",
    )
    return fig


def weather_lapse():
    ex, kr = _weather_tables()
    head = load_json(WX, "meta")["headline"]["physics"]
    rows = [
        (
            f"{SHORT[a]} (low stations)",
            float(ex[ex.arm == a].dTdz.median()),
            None,
            P.ARM_COLOR[a],
        )
        for a in COMPETITORS
    ]
    rows.append(
        (
            "universal kriging (low)",
            float(kr[kr.arm == "uk"].dTdz.iloc[0]),
            None,
            P.INK_2,
        )
    )
    rows.append(
        (
            "physics, all stations",
            head["dTdz"],
            head["sigma"]["Gamma"],
            P.ARM_COLOR["physics"],
        )
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (_lab, v, s, col) in enumerate(rows):
        ax.errorbar(v, i, xerr=s, fmt="o", color=col, ms=8, capsize=3)
    ax.axvline(LAPSE_STANDARD, color=P.ARM_COLOR["oracle"], ls="--", lw=1.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1.2)
    ax.text(LAPSE_STANDARD, -0.9, " standard atmosphere", color=P.INK_2, fontsize=9)
    P._style(ax, "Recovered lapse rate", "dT/dz [K/km]")
    return fig


def weather_conclusion() -> str:
    ex, kr = _weather_tables()
    m = _med(ex, "nrmse_out")
    uk = float(kr[kr.arm == "uk"].nrmse_out.iloc[0])
    head = load_json(WX, "meta")["headline"]
    ph = head["physics"]
    return (
        f"Trained below {ex.z_max_train_km.iloc[0]:.1f} km, nRMSE on the high stations: "
        f"universal kriging {_fmt(uk)}, SR {_fmt(m.sr)}, physics {_fmt(m.physics)}, "
        f"PINN {_fmt(m.pinn)}, NN {_fmt(m.nn)}; lapse rate from all stations "
        f"{ph['dTdz']:.2f} +/- {ph['sigma']['Gamma']:.2f} K/km against "
        f"{LAPSE_STANDARD} ({head['gamma_minus_published_sigma']:+.1f} sigma)."
    )


# ---------------------------------------------------------------------------
# g. constants recovered by the physics arm
# ---------------------------------------------------------------------------

CONSTANTS = [
    ("Rydberg R vs Bohr prediction", "hydrogen R vs Bohr value"),
    ("GR coefficient alpha (Mercury)", "Mercury GR coefficient alpha"),
    ("GM_sun from Kepler", "GM_sun from Kepler's third law"),
    ("CMB temperature T", "T_CMB from the FIRAS spectrum"),
    ("pulsar braking index n", "pulsar braking index n vs 3"),
]


def _constants() -> pd.DataFrame:
    rec = load_json("", "headline")["parameter_recovery"]
    rows = []
    for key, label in CONSTANTS:
        r = next((r for r in rec if r["quantity"].startswith(key)), None)
        if r is None or r["recovered"] is None:
            continue
        pub = float(r["published"])
        rel = (float(r["recovered"]) - pub) / abs(pub)
        sig = float(r["sigma"]) / abs(pub) if r["sigma"] else np.nan
        note = " (partly by construction)" if "construction" in r["deviation"] else ""
        rows.append(
            {
                "label": label,
                "rel": rel,
                "rel_sigma": sig,
                "n_sigma": rel / sig,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def constants_result():
    c = _constants()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(c))
    ax.barh(y, c.rel.abs(), height=0.55, color=P.ARM_COLOR["physics"])
    ax.plot(c.rel_sigma, y, "|", color=P.INK, ms=16, mew=2)
    for i, r in c.iterrows():
        ax.text(
            abs(r.rel) * 1.3, i, f"{r.rel:+.1e}", va="center", color=P.INK_2, fontsize=9
        )
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(c.label)
    ax.invert_yaxis()
    ax.set_xlim(right=max(c.rel.abs().max(), c.rel_sigma.max()) * 20)
    ax.grid(axis="y", visible=False)
    P._style(
        ax,
        "Physics arm: deviation from published (tick: 1 sigma)",
        "|recovered - published| / published (log)",
    )
    return fig


def constants_conclusion() -> str:
    c = _constants()
    return (
        "Relative deviation (fit sigmas): "
        + "; ".join(
            f"{r.label} {r.rel:+.1e} ({r.n_sigma:+.1f} sigma){r.note}"
            for r in c.itertuples()
        )
        + "."
    )


# ---------------------------------------------------------------------------
# h. the amount of training data (H7)
# ---------------------------------------------------------------------------

BUDGET_TRACKS = [
    "quantum/hydrogen",
    "quantum/cmb",
    "quantum/helium",
    "fields/weather/july",
    "gravity/pulsar_spindown/sim",
]
BUDGET_ARMS = {
    "physics": ("physics", P.ARM_COLOR["physics"]),
    "nn": ("NN", P.ARM_COLOR["nn"]),
    "pinn": ("PINN, one weight", P.ARM_COLOR["pinn"]),
    "pinn_budget_w": ("PINN, weight per size", P.ARM_COLOR["sr"]),
}


def _budget_tables() -> dict[str, pd.DataFrame]:
    out = {}
    for track in BUDGET_TRACKS:
        p = _results(track, "budget_weight.csv")
        if p.exists():
            out[track] = pd.read_csv(p)
    return out


def _budget_missing(tables) -> str | None:
    missing = [t for t in BUDGET_TRACKS if t not in tables]
    if not missing:
        return None
    return "Not yet produced (results/<track>/budget_weight.csv): " + ", ".join(missing)


def _chosen_weight(track: str, t: pd.DataFrame) -> pd.Series:
    if "w_phys" in t:
        w = t[t.arm == "pinn_budget_w"].groupby("n_train").w_phys.first()
        if w.notna().all() and len(w):
            return w
    sel = load_table(track, "budget_w_phys_selection")
    return sel[sel.chosen].groupby("n_train").w_phys.first()


def budget_result():
    tables = _budget_tables()
    note = _budget_missing(tables)
    if note:
        print(note)
    if not tables:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()
    for ax, (track, t) in zip(axes, tables.items(), strict=False):
        g = t.groupby(["arm", "n_train"]).nrmse_out.median()
        for arm, (label, color) in BUDGET_ARMS.items():
            if arm in g.index.get_level_values(0):
                s = g.loc[arm]
                ax.plot(s.index, s.values, "o-", color=color, ms=5, label=label)
        _log_axes(ax)
        P._style(ax, track, "training points (log)", "held-out nRMSE (log)")
    for ax in axes[len(tables) :]:
        ax.axis("off")
    h, lab = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        h,
        lab,
        loc="center",
        fontsize=10,
        labelcolor=P.INK_2,
        title=LOWER,
        title_fontsize=9,
    )
    axes[-1].axis("off")
    return fig


def _log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    # 1-2-5 ticks: budgets span barely a decade, so decade ticks alone leave
    # some panels with a single labelled tick
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def budget_weight_result():
    tables = _budget_tables()
    if not tables:
        print(_budget_missing(tables))
        return None
    # one panel per track: five weight paths on one axes cross too often to read
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    axes = axes.ravel()
    for ax, (track, t) in zip(axes, tables.items(), strict=False):
        w = _chosen_weight(track, t)
        ax.plot(w.index, w.values, "o-", color=P.ARM_COLOR["sr"], ms=5)
        _log_axes(ax)
        P._style(ax, track, "training points (log)", "chosen w_phys (log)")
    for ax in axes[len(tables) :]:
        ax.axis("off")
    fig.suptitle(
        "Physics weight chosen at each training-set size (tuning seeds)",
        color=P.INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return fig


def budget_conclusion() -> str:
    tables = _budget_tables()
    if not tables:
        return _budget_missing(tables) or ""
    cells = better_phys = total = shrink = 0
    for track, t in tables.items():
        med = t.groupby(["n_train", "arm"]).nrmse_out.median().unstack()
        ok = med.dropna(subset=["pinn", "pinn_budget_w"])
        cells += int((ok.pinn_budget_w <= ok.pinn).sum())
        better_phys += int((ok.pinn_budget_w < ok.physics).sum())
        total += len(ok)
        w = _chosen_weight(track, t)
        shrink += int(w.iloc[0] >= w.iloc[-1])
    text = (
        f"Weight per size is no worse than the single weight in {cells}/{total} "
        f"(track, size) cells and beats physics in {better_phys}/{total}; the weight "
        f"at the smallest size is at least that at the largest on {shrink}/{len(tables)} "
        "tracks."
    )
    note = _budget_missing(tables)
    if note:
        return f"{text} {note}."
    from physprior.benchmark.budget_weight import verdicts

    refuted = verdicts()["refuted"]
    return f"{text} H7 by its pre-registered criterion: {'refuted' if refuted else 'not refuted'}."


# ---------------------------------------------------------------------------
# the registry and `show`
# ---------------------------------------------------------------------------


@dataclass
class Task:
    title: str
    goal: str
    data: list[Callable] = field(default_factory=list)
    results: list[Callable] = field(default_factory=list)
    conclusion: Callable[[], str] = lambda: ""


TASKS: dict[str, Task] = {
    "gw150914": Task(
        "Extrapolating the GW150914 chirp",
        "Six cycles of the frequency track of LIGO's first detection. Train on the "
        "earlier cycles, predict the later ones, where the frequency rises fastest.",
        [gw150914_data],
        [gw150914_result],
        gw150914_conclusion,
    ),
    "neglected": Task(
        "When a law with a missing term still helps",
        "Simulations where the truth is a law plus a term the model does not contain. "
        "Here a pendulum whose damping the law leaves out.",
        [neglected_data],
        [neglected_result],
        neglected_conclusion,
    ),
    "pulsars": Task(
        "Pulsar braking index",
        "The magnetic-dipole law predicts n = 3 for every pulsar. Test it on the "
        "measured young pulsars, then check on a simulated control whose n falls "
        "with age that the PINN could learn such a dependence.",
        [pulsars_data],
        [pulsars_result],
        pulsars_conclusion,
    ),
    "reconstruction": Task(
        "Reconstructing a field from sparse sensors in 1-D, 2-D and 3-D",
        "A Poisson field from three sources, sampled by N noisy sensors. How many "
        "sensors does each method need as the dimension grows?",
        [reconstruction_data],
        [reconstruction_result],
        reconstruction_conclusion,
    ),
    "dynamics": Task(
        "Learning the update rule of a dynamical system",
        "A stepper learns u(t + dt) from pairs of states and is then iterated far "
        "beyond its training trajectories.",
        [dynamics_data],
        [dynamics_result_energy, dynamics_result_horizon],
        dynamics_conclusion,
    ),
    "weather": Task(
        "Temperature over the Alps",
        "July mean temperature at weather stations. Train on the low stations, "
        "predict the high ones, and read off the lapse rate.",
        [weather_data],
        [weather_result, weather_lapse],
        weather_conclusion,
    ),
    "constants": Task(
        "Constants recovered by the physics arm",
        "The fitted law returns a constant with an error bar, which can be compared "
        "with the published value.",
        [],
        [constants_result],
        constants_conclusion,
    ),
    "data_budget": Task(
        "The amount of training data (H7)",
        "Error against training-set size on five tracks, with the PINN's physics "
        "weight either fixed per track or chosen again for each size.",
        [],
        [budget_result, budget_weight_result],
        budget_conclusion,
    ),
}


def show(task: str) -> None:
    """Draw one task's data and result plots, then print its conclusion."""
    from IPython.display import display

    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choose one of {list(TASKS)}")
    t = TASKS[task]
    print(t.title)
    for fn in [*t.data, *t.results]:
        fig = fn()
        if fig is not None:
            display(fig)
            plt.close(fig)
    print(t.conclusion())


# ---------------------------------------------------------------------------
# the notebook
# ---------------------------------------------------------------------------

SETUP = """import os, warnings
warnings.filterwarnings("ignore")
# nothing here runs symbolic regression, so skip the Julia bootstrap
os.environ.setdefault("PHYSPRIOR_NO_JULIA", "1")
import matplotlib.pyplot as plt
from physprior.viz import plots as P
from physprior.reporting import summary as S
P.use_style()
"""

INTRO = """
# physprior: summary of results

This project asks what a physics prior buys a machine-learning model, and what
it costs when the law is wrong. Models that know a law (a fitted law, a
physics-informed network, symbolic regression) are compared with a tuned black
box on the same splits. The comparison runs on real measurements and on
simulations whose truth is known, so each method's own error can be told apart
from a flaw in the law.
"""

METHOD = """
## Method

The arms:

| arm | what it is |
|---|---|
| `oracle` | the published law with published constants; a reference, not a competitor |
| `physics` | the law with its constants fitted, with error bars |
| `pinn` | the law plus a learned correction (or an ODE residual), constants trainable |
| `sr` | symbolic regression, given no law |
| `nn` | a tuned multilayer perceptron, given no law |

The PINN comes in two shapes. Where the law is a differential equation, the
network is the solution and the law enters as a residual (A). Where the law is
algebraic, the network is a correction added to the law and penalised toward
zero (B).

![the two PINN shapes](../figures/summary/pinn_anatomy.png)

The physics weight `w_phys` sets how strongly the correction is held to zero.
It is chosen per track on the tuning seeds, from a validation block at the top
of the training range, with ties going to the larger weight. Loss balancing,
which annealed the weight upward until the PINN reduced to the fitted law, was
removed from the default ([DECISIONS.md](../docs/DECISIONS.md)).

Protocol:

- every choice is tuned on seeds 3, 7, 19; every reported number is on seeds 11, 23, 42;
- the predictions for each study were written before its runs ([HYPOTHESES.md](../docs/HYPOTHESES.md));
- every number in this notebook is read from `results/` when it runs; none is typed.
"""

DOCS = {
    "gw150914": "relativity/README.md",
    "neglected": "neglected/README.md",
    "pulsars": "gravity/README.md",
    "reconstruction": "reconstruction/README.md",
    "dynamics": "dynamics/README.md",
    "weather": "fields/weather.md",
    "constants": "RESULTS.md",
    "data_budget": "HYPOTHESES.md",
}

IN_PROGRESS = """
## In progress

- **Helium.** With its tuned weight, the PINN extrapolates worse than the
  fitted hydrogenic law, so H3 is refuted there
  ([quantum](../docs/quantum/README.md#the-real-data-track-quantumhelium)).
- **Algebraic tracks with the tuned PINN.** With the correction active, it
  extrapolates worse than the fitted law
  ([DECISIONS.md](../docs/DECISIONS.md), [optimization §5](../docs/optimization/README.md)).
- **Helmholtz-residual PINN on the radio field.** The receivers record no phase,
  so the residual leaves the field under-determined
  ([rf](../docs/fields/rf.md#shape-a-a-helmholtz-residual-pinn-on-a-window)).
- **3-D PDE-residual PINN in reconstruction.** Run at two sensor counts per
  dimension with an untuned weight; in 3-D it is less accurate than a plain GP
  at the same count ([reconstruction](../docs/reconstruction/README.md)).
"""


def _section(i: int, key: str) -> list:
    t = TASKS[key]
    letter = "abcdefgh"[i]
    cells = [
        md(
            f"## {letter}. {t.title}\n\n{t.goal} "
            f"Details: [docs/{DOCS[key]}](../docs/{DOCS[key]})."
        )
    ]
    for fn in t.data:
        cells.append(code(f"S.{fn.__name__}();"))
    for fn in t.results:
        cells.append(code(f"S.{fn.__name__}();"))
    cells.append(code(f"print(S.{t.conclusion.__name__}())"))
    return cells


def notebook():
    keys = list(TASKS)
    cells = [md(INTRO), code(SETUP), md(METHOD)]
    cells += [
        md(
            "## Choose a task\n\nEdit `TASK` and re-run the cell to see one task's "
            "data, result and conclusion."
        ),
        code(f'TASK = "gw150914"  # one of: {", ".join(keys)}\nS.show(TASK)'),
        md(
            "## Results by task\n\nEach section shows the data, then the result, then "
            "a one-line conclusion computed from `results/`."
        ),
    ]
    for i, key in enumerate(keys):
        cells += _section(i, key)
    cells.append(md(IN_PROGRESS))
    return _nb(cells, "physprior: summary of results")


def build(execute: bool = False) -> nbf.NotebookNode:
    """Write notebooks/summary.ipynb, executing it with the `physprior` kernel."""
    settings = get_settings()
    settings.notebooks_dir.mkdir(parents=True, exist_ok=True)
    nb = notebook()
    path = settings.notebooks_dir / "summary.ipynb"
    if execute:
        from nbclient import NotebookClient

        NotebookClient(
            nb,
            timeout=600,
            kernel_name="physprior",
            resources={"metadata": {"path": str(settings.notebooks_dir)}},
            allow_errors=True,
        ).execute()
        errs = [
            c
            for c in nb.cells
            if any(o.get("output_type") == "error" for o in c.get("outputs", []))
        ]
        print(f"executed, {len(errs)} cell(s) with errors")
    nbf.write(nb, path)
    print(f"wrote {path}")
    return nb


if __name__ == "__main__":
    build(execute="--execute" in sys.argv)
