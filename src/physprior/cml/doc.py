"""docs/cml/README.md and figures/cml/, generated from results/cml.

Every number on the page is read from `results/cml/*.csv`. Regenerate with
`physprior cml --doc-only`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import wilcoxon

from physprior.config import get_settings
from physprior.viz.palette import ARM_PALETTE
from physprior.viz.plots import INK_MUTED, SURFACE, _style, use_style

DOC = "cml/README.md"
TRACK = "cml"
LABEL = {
    "pl_itu": "power law, ITU k and alpha",
    "pl_cal": "power law, calibrated",
    "gru": "GRU (data only)",
    "hybrid_joint": "hybrid, joint",
    "hybrid_gate": "hybrid, gate only",
    "hybrid_phased": "hybrid, phased",
}
COLOR = {
    "pl_itu": INK_MUTED,
    "pl_cal": ARM_PALETTE[0],
    "gru": ARM_PALETTE[1],
    "hybrid_joint": ARM_PALETTE[3],
    "hybrid_gate": ARM_PALETTE[3],
    "hybrid_phased": ARM_PALETTE[2],
}
DATA_LABEL = {
    "openmrg": "OpenMRG (Gothenburg)",
    "openrainer": "OpenRainER (Emilia-Romagna)",
    "netherlands": "Netherlands",
    "openmesh": "OpenMesh (New York)",
    "sim_iid": "sim: i.i.d. rain",
    "sim_ar1": "sim: AR(1) rain",
    "sim_iid_waa": "sim: i.i.d. + wet antenna",
    "sim_ar1_waa": "sim: AR(1) + wet antenna",
}
HYBRIDS = ("hybrid_joint", "hybrid_gate", "hybrid_phased")


def _dir():
    return get_settings().results_dir / TRACK


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _dir()
    return pd.read_csv(d / "runs.csv"), pd.read_csv(d / "datasets.csv")


def _f(x, nd=3) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.{nd}g}"


def _plain_log(ax, axes: str = "xy") -> None:
    """Log axes labelled with plain numbers and no minor labels."""
    for a in axes:
        axis = ax.xaxis if a == "x" else ax.yaxis
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())


def _save(fig, name: str):
    p = get_settings().figures(TRACK) / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over report seeds of each metric, per dataset and arm."""
    m = runs[runs["stage"] == "main"]
    cols = [
        "nrmse",
        "nrmse_link_median",
        "nbias",
        "corr",
        "nrmse_wet",
        "gate_mean",
        "gate_wet",
    ]
    cols = [c for c in cols if c in m]
    return m.groupby(["dataset", "arm"], sort=False)[cols].mean().reset_index()


def best_hybrid_vs(runs: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the phased hybrid against the better of its branches
    (pl_cal, gru), seed by seed."""
    m = runs[runs["stage"] == "main"]
    rows = []
    for d, g in m.groupby("dataset", sort=False):
        p = g.pivot_table(index="seed", columns="arm", values="nrmse")
        if not {"hybrid_phased", "pl_cal", "gru"} <= set(p.columns):
            continue
        best_branch = np.minimum(p["pl_cal"], p["gru"])
        rows.append(
            {
                "dataset": d,
                "pl_itu": p["pl_itu"].mean() if "pl_itu" in p else np.nan,
                "pl_cal": p["pl_cal"].mean(),
                "gru": p["gru"].mean(),
                "hybrid_phased": p["hybrid_phased"].mean(),
                "gain_vs_gru_pct": 100 * (1 - (p["hybrid_phased"] / p["gru"]).mean()),
                "gain_vs_best_branch_pct": 100
                * (1 - (p["hybrid_phased"] / best_branch).mean()),
                "seeds_beating_best_branch": int(
                    (p["hybrid_phased"] < best_branch).sum()
                ),
                "n_seeds": len(p),
            }
        )
    return pd.DataFrame(rows)


def fig_main(runs: pd.DataFrame):
    use_style()
    t = main_table(runs)
    data = list(dict.fromkeys(t["dataset"]))
    arms = [a for a in LABEL if a in set(t["arm"])]
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(data) + 1.5))
    off = np.linspace(-0.3, 0.3, len(arms))
    for k, a in enumerate(arms):
        g = t[t["arm"] == a].set_index("dataset")
        y = [len(data) - 1 - i + off[k] for i in range(len(data))]
        v = [g["nrmse"].get(d, np.nan) for d in data]
        ax.scatter(
            v,
            y,
            s=46,
            color=COLOR[a],
            edgecolor=SURFACE,
            lw=1.2,
            marker="o" if a.startswith("hybrid") else "s",
            label=LABEL[a],
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yticks(range(len(data)), [DATA_LABEL.get(d, d) for d in data][::-1])
    _style(ax, xlabel="NRMSE on test days (log scale), mean of seeds 11, 23, 42")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    return _save(fig, "main_nrmse")


def fig_budget(runs: pd.DataFrame):
    use_style()
    b = runs[runs["stage"].isin(["budget", "main"])]
    data = list(dict.fromkeys(runs.loc[runs["stage"] == "budget", "dataset"]))
    if not data:
        return None
    fig, axes = plt.subplots(
        1, len(data), figsize=(3.1 * len(data), 3.2), squeeze=False
    )
    for ax, d in zip(axes[0], data, strict=True):
        g = b[b["dataset"] == d]
        for a in ("pl_cal", "gru", "hybrid_gate", "hybrid_phased"):
            h = g[g["arm"] == a].groupby("budget")["nrmse"].mean()
            if h.empty:
                continue
            ax.plot(
                100 * h.index,
                h.values,
                marker="o",
                lw=2,
                color=COLOR[a],
                ls="--" if a == "hybrid_gate" else "-",
                label=LABEL[a],
            )
        ax.set_xscale("log")
        _style(ax, title=DATA_LABEL.get(d, d), xlabel="training days used (%)")
    axes[0][0].set_ylabel("NRMSE")
    axes[0][0].legend(frameon=False, fontsize=7)
    return _save(fig, "budget")


def fig_gate_noise(runs: pd.DataFrame):
    use_style()
    n = runs[(runs["stage"] == "noise") & (runs["arm"] == "hybrid_phased")]
    if n.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for d, ls in (("sim_iid", "--"), ("sim_ar1_waa", "-")):
        g = n[n["dataset"] == d].sort_values("noise_db")
        ax.plot(
            g["noise_db"],
            g["gate_wet"],
            marker="o",
            lw=2,
            ls=ls,
            color=COLOR["hybrid_phased"],
            label=DATA_LABEL[d],
        )
    ax.set_xscale("log")
    _plain_log(ax, "x")
    ax.set_ylim(0, 1)
    _style(ax, xlabel="noise sd (dB)", ylabel="mean gate on wet bins (1 = power law)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "gate_noise")


def fig_intensity(it: pd.DataFrame):
    """Mean estimate against mean reference in each intensity class."""
    use_style()
    data = list(dict.fromkeys(it["dataset"]))
    ncol = 4
    nrow = int(np.ceil(len(data) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow), squeeze=False
    )
    for ax, d in zip(axes.ravel(), data, strict=False):
        g = it[(it["dataset"] == d) & (it["class"] != "dry")]
        for a in ("pl_cal", "gru", "hybrid_phased"):
            h = g[g["arm"] == a]
            ax.plot(
                h["mean_ref"],
                h["mean_est"],
                marker="o",
                lw=2,
                color=COLOR[a],
                label=LABEL[a],
            )
        lim = [g["mean_ref"].min() * 0.7, g["mean_ref"].max() * 1.4]
        ax.plot(lim, lim, color=INK_MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax)
        _style(ax, title=DATA_LABEL.get(d, d), xlabel="reference (mm/h)")
    for ax in axes.ravel()[len(data) :]:
        ax.set_visible(False)
    axes[0][0].set_ylabel("estimate (mm/h)")
    axes[0][0].legend(frameon=False, fontsize=7)
    return _save(fig, "intensity")


def _table(df: pd.DataFrame, cols: list[str], fmt=None) -> list[str]:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append(
            "| "
            + " | ".join(
                fmt.get(c, lambda v: _f(v) if isinstance(v, float) else str(v))(r[c])
                for c in cols
            )
            + " |"
        )
    return out


def render_doc() -> str:
    runs, info = load()
    fig_main(runs)
    fig_budget(runs)
    fig_gate_noise(runs)
    t = main_table(runs)
    L = [
        "# Rain from microwave links: power law, network and hybrid",
        "",
        "<!-- Generated by physprior.cml.doc.render_doc() from results/cml. "
        "Do not edit by hand. -->",
        "",
        "A commercial microwave link loses signal in rain, A = k R^alpha L "
        "(ITU-R P.838-3). Three ways to turn that loss into a rain rate are "
        "compared on four link archives and four simulations: the power law "
        "(with ITU coefficients, or calibrated per link), a GRU trained on "
        "gauge or radar labels, and the gated hybrid of Jacoby et al. (ICASSP "
        "2026), where a learned gate blends the two. Setup, splits and results "
        "are built here from scratch; the plan is `docs/plans/CML_PLAN.md`, the "
        "code `src/physprior/cml/`, and `physprior cml` reruns it.",
        "",
        "## Data",
        "",
        *_table(
            info.assign(dataset=info["dataset"].map(lambda d: DATA_LABEL.get(d, d))),
            [
                "dataset",
                "n_links",
                "n_samples",
                "wet_fraction",
                "mean_rain_mmh",
                "freq_min",
                "freq_max",
                "len_min",
                "len_max",
            ],
        ),
        "",
        "A sample is one bin of max(10 min, reference step): 10 min on OpenMRG and the simulations, 15 min on OpenRainER and OpenMesh, 60 min in the Netherlands. The "
        "input is the excess attenuation over a causal 24-h rolling median, "
        "over the 30 min that end with the bin. Days are split 70/15/15 into "
        "train, validation and test, the same split for every arm. Links whose "
        "label-free noise sd exceeds max(0.6 dB, 1.5 quantisation steps) are dropped "
        "(`results/cml/dropped_links.csv`).",
        "",
        "## Main comparison",
        "",
        "NRMSE = RMSE / mean(r) and NBIAS = mean(r_hat - r) / mean(r) on every "
        "test bin; `nrmse_link_median` is the median over links of the "
        "per-link NRMSE, which a single bad link cannot dominate; `wet` "
        "columns use bins above 0.1 mm/h. Mean of seeds 11, 23, 42. `gate` is "
        "the mean weight on the power-law branch.",
        "",
    ]
    for d in dict.fromkeys(t["dataset"]):
        g = t[t["dataset"] == d].copy()
        g["arm"] = g["arm"].map(LABEL)
        cols = ["arm", "nrmse", "nrmse_link_median", "nbias", "corr", "nrmse_wet"]
        if "gate_mean" in g:
            cols += ["gate_mean", "gate_wet"]
        L += [f"### {DATA_LABEL.get(d, d)}", "", *_table(g, cols), ""]
    bh = best_hybrid_vs(runs)
    if not bh.empty:
        bh2 = bh.assign(dataset=bh["dataset"].map(lambda d: DATA_LABEL.get(d, d)))
        L += [
            "## The phased hybrid against its branches",
            "",
            "Gain in NRMSE, in percent, of the phased hybrid over the GRU and over "
            "the better of the calibrated power law and the GRU, per seed then "
            "averaged; the last column counts seeds where the hybrid beats both.",
            "",
            *_table(
                bh2,
                [
                    "dataset",
                    "pl_itu",
                    "pl_cal",
                    "gru",
                    "hybrid_phased",
                    "gain_vs_gru_pct",
                    "gain_vs_best_branch_pct",
                    "seeds_beating_best_branch",
                    "n_seeds",
                ],
            ),
            "",
        ]
        m = runs[runs["stage"] == "main"]
        p = m.pivot_table(index=["dataset", "seed"], columns="arm", values="nrmse")
        if {"hybrid_phased", "gru", "pl_cal"} <= set(p.columns):
            for base in ("gru", "pl_cal"):
                diff = np.log(p["hybrid_phased"] / p[base]).dropna().to_numpy()
                pv = (
                    wilcoxon(diff).pvalue if len(diff) >= 6 and np.any(diff) else np.nan
                )
                L.append(
                    f"Over all {len(diff)} (dataset, seed) pairs the phased hybrid "
                    f"has lower NRMSE than `{base}` in {(diff < 0).sum()} "
                    f"(Wilcoxon p = {_f(pv, 2)})."
                )
            L.append("")
    L += ["![main](../../figures/cml/main_nrmse.png)", ""]
    adir = _dir()
    if (adir / "intensity.csv").exists():
        it = pd.read_csv(adir / "intensity.csv")
        det = pd.read_csv(adir / "detection.csv")
        ev = pd.read_csv(adir / "events.csv")
        fig_intensity(it)
        arms = ["pl_cal", "gru", "hybrid_phased"]
        L += [
            "## By rain intensity",
            "",
            "Test bins of the three seeds pooled, split by the reference rate "
            "(mm/h): dry < 0.1, light 0.1-1, moderate 1-5, heavy 5-20, very "
            "heavy > 20. `rel_bias` is mean(estimate - reference) / "
            "mean(reference) in the class; in the dry class it is the mean "
            "estimate itself, in mm/h, since the reference mean is near zero.",
            "",
        ]
        for d in dict.fromkeys(it["dataset"]):
            g = it[(it["dataset"] == d) & it["arm"].isin(arms)]
            piv = g.pivot_table(
                index="class", columns="arm", values="rel_bias", sort=False
            )
            dry = g[g["class"] == "dry"].set_index("arm")["mean_est"]
            rows = []
            for c in [
                c
                for _, _, c in (
                    (0, 0, "dry"),
                    (0, 0, "light"),
                    (0, 0, "moderate"),
                    (0, 0, "heavy"),
                    (0, 0, "very heavy"),
                )
                if c in piv.index
            ]:
                n = int(g[(g["class"] == c)]["bins"].iloc[0])
                vals = [
                    dry.get(a, np.nan) if c == "dry" else piv.loc[c].get(a, np.nan)
                    for a in arms
                ]
                rows.append(
                    {
                        "class": c,
                        "bins": n,
                        **{LABEL[a]: v for a, v in zip(arms, vals, strict=True)},
                    }
                )
            L += [
                f"### {DATA_LABEL.get(d, d)}",
                "",
                *_table(
                    pd.DataFrame(rows), ["class", "bins", *[LABEL[a] for a in arms]]
                ),
                "",
            ]
        L += [
            "![intensity](../../figures/cml/intensity.png)",
            "",
            "### Detection at 0.1 mm/h",
            "",
            "POD: fraction of wet bins estimated wet. FAR: fraction of bins "
            "estimated wet that were dry. `dry called wet`: fraction of dry "
            "bins estimated wet.",
            "",
        ]
        dd = det[det["arm"].isin(arms)].copy()
        dd["dataset"] = dd["dataset"].map(lambda d: DATA_LABEL.get(d, d))
        dd["arm"] = dd["arm"].map(LABEL)
        dd = dd.rename(columns={"dry_called_wet": "dry called wet"})
        L += [*_table(dd, ["dataset", "arm", "pod", "far", "dry called wet"]), ""]
        L += [
            "### Rain events",
            "",
            "An event is a run of wet reference bins on one link with gaps of "
            "at most 60 min. Median relative error of the event total (and its "
            "median absolute value) and of the event peak, by event total.",
            "",
        ]
        ee = ev[ev["arm"].isin(arms)].copy()
        ee["dataset"] = ee["dataset"].map(lambda d: DATA_LABEL.get(d, d))
        ee["arm"] = ee["arm"].map(LABEL)
        L += [
            *_table(
                ee,
                [
                    "dataset",
                    "event_class",
                    "arm",
                    "events",
                    "total_err_median",
                    "total_abs_err_median",
                    "peak_err_median",
                ],
            ),
            "",
        ]
    if (runs["stage"] == "budget").any():
        L += [
            "## Data efficiency",
            "",
            "The same test days, training on 5, 15, 40 and 100 % of the training "
            "days (every k-th day in rain order, so wet and dry days are kept in "
            "proportion).",
            "",
            "![budget](../../figures/cml/budget.png)",
            "",
        ]
    if (runs["stage"] == "noise").any():
        L += [
            "## What the gate does with noise",
            "",
            "Simulated links, noise sd from 0.05 to 0.8 dB: the mean gate of the "
            "phased hybrid on wet bins (1 = all power law).",
            "",
            "![gate_noise](../../figures/cml/gate_noise.png)",
            "",
        ]
    text = "\n".join(L).rstrip() + "\n"
    p = get_settings().root / "docs" / DOC
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return text
