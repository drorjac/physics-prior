"""docs/theory/*.md, generated from results/symbolic/.

    render_doc()  ->  docs/theory/symbolic_regression.md, docs/theory/packages.md

The prose describes method and is fixed; every number in these pages is read
from a results file here, so a re-run regenerates the pages rather than
leaving them to drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json, load_table

AREA = "symbolic"
MISSING = "_(not yet run: `physprior.symbolic.studies.run()`)_"


def _t(name: str) -> pd.DataFrame | None:
    try:
        return load_table(AREA, name)
    except FileNotFoundError:
        return None


def _j(name: str):
    try:
        return load_json(AREA, name)
    except FileNotFoundError:
        return None


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (int, np.integer)) or (
                isinstance(v, (float, np.floating))
                and np.isfinite(v)
                and float(v).is_integer()
                and abs(float(v)) < 1e9
            ):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                s = str(v).replace("|", "\\|")
                cells.append(f"`{s}`" if c in ("expression", "vocabulary") else s)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _code_expr(e: str, n: int = 70) -> str:
    return e if len(e) <= n else e[: n - 3] + "..."


# ---------------------------------------------------------------------------
# result blocks
# ---------------------------------------------------------------------------


def _block_growth() -> str:
    counts = _t("growth_counts")
    pruned = _t("growth_pruned")
    depth = _t("growth_depth")
    ttf = _t("time_to_find")
    voc = _t("time_vs_vocabulary")
    if counts is None:
        return MISSING
    parts = []
    piv = counts.pivot_table(index="size", columns="n_ops", values="n_trees_raw")
    piv = piv[[c for c in piv.columns if c in (2, 4, 6, 9, 11)]]
    piv.columns = [f"{c} operators" for c in piv.columns]
    parts.append(
        "Distinct trees of each size, two leaf types (the variable and a "
        "constant), from the recurrence (`growth_counts.csv`):\n\n"
        + _md(piv.reset_index(), ".3g")
    )
    if pruned is not None:
        p = pruned.copy()
        p["kept"] = p["n_trees_pruned"] / p["n_trees_raw"]
        parts.append(
            "Removing syntactic redundancy (operators on constants only, "
            "commutative duplicates, `exp(log(.))`, more than two constants) "
            "before fitting (`growth_pruned.csv`):\n\n"
            + _md(p[["vocabulary", "size", "n_trees_raw", "n_trees_pruned", "kept"]])
        )
    if depth is not None:
        parts.append(
            "By depth instead of size the growth is doubly exponential, "
            "$D(d) = L + U D(d-1) + B D(d-1)^2$ (`growth_depth.csv`, nine "
            "operators):\n\n" + _md(depth, ".3g")
        )
    if ttf is not None:
        g = (
            ttf.groupby("law")
            .agg(
                seeds=("seed", "count"),
                size=("size_reached", "max"),
                trees=("n_evaluated", "mean"),
                seconds=("seconds", "mean"),
                recovered=("recovered", "sum"),
            )
            .reset_index()
        )
        parts.append(
            "Exhaustive search to the noise floor on noise-free data, 24 "
            "points, nine operators (`time_to_find.csv`; seconds are wall "
            "time on a shared machine, so read them as relative):\n\n" + _md(g)
        )
    if voc is not None:
        v = voc[["n_ops", "n_evaluated", "seconds", "recovered"]]
        parts.append(
            "The same target (Bohr, size 6) as the vocabulary grows "
            "(`time_vs_vocabulary.csv`):\n\n" + _md(v)
        )
    parts.append("![search space](../../figures/symbolic/growth.png)")
    return "\n\n".join(parts)


def _recovery_frames():
    noise = [d for d in (_t("recovery_noise"), _t("recovery_pysr")) if d is not None]
    noise_df = pd.concat(noise, ignore_index=True) if noise else None
    budget = _t("recovery_budget")
    if budget is not None and noise_df is not None:
        extra = noise_df[(noise_df.noise == 0.01) & (noise_df.method != "pysr")]
        budget = pd.concat([budget, extra], ignore_index=True)
    return noise_df, budget


def _rate_table(df: pd.DataFrame, col: str) -> str:
    g = (
        df.groupby(["law", "method", col])
        .agg(
            recovered=("recovered", "sum"),
            on_front=("on_front", "sum"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    g["recovered"] = (
        g["recovered"].astype(int).astype(str) + "/" + g["runs"].astype(str)
    )
    g["on_front"] = g["on_front"].astype(int).astype(str) + "/" + g["runs"].astype(str)
    piv = g.pivot_table(
        index=["law", "method"], columns=col, values="recovered", aggfunc="first"
    )
    piv.columns = [f"{col} = {c:g}" for c in piv.columns]
    return _md(piv.reset_index())


def _block_recovery() -> str:
    noise, budget = _recovery_frames()
    tune = _j("tune_gp")
    if noise is None:
        return MISSING
    parts = []
    if tune:
        rates = ", ".join(
            f"{float(k):g}: {v:.2f}"
            for k, v in sorted(tune["rate"].items(), key=lambda kv: float(kv[0]))
        )
        parts.append(
            f"GP parsimony was chosen on the tuning seeds {tune['seeds']} at 1 % "
            f"noise, 24 points: recovery rate by parsimony {rates}; chosen "
            f"**{tune['chosen_parsimony']:g}** (`tune_gp.json`)."
        )
    parts.append(
        "Recovery vs relative noise, 24 points, reporting seeds 11/23/42 "
        "(`recovery_noise.csv`, `recovery_pysr.csv`). Each cell is "
        "recovered/runs for the **selected** expression:\n\n"
        + _rate_table(noise, "noise")
    )
    common = noise[noise.law != "planck"]
    front = (
        common.groupby("method")
        .agg(selected=("recovered", "mean"), on_front=("on_front", "mean"))
        .reset_index()
    )
    parts.append(
        "Selected vs anywhere on the front, pooled over the noise levels and "
        "the three laws every method was run on (Planck excluded, since "
        "exhaustive search was not). The difference between the columns is "
        "what the selection rule loses:\n\n" + _md(front, ".2f")
    )
    if budget is not None:
        parts.append(
            "Recovery vs number of samples at 1 % noise (`recovery_budget.csv`; "
            "the n = 24 column is the noise sweep's):\n\n" + _rate_table(budget, "n")
        )
    t = (
        noise.groupby("method")
        .agg(median_seconds=("seconds", "median"), runs=("seed", "count"))
        .reset_index()
    )
    parts.append("Median wall time per run:\n\n" + _md(t, ".3g"))
    failed = noise[(~noise.recovered.astype(bool)) & (noise.noise <= 0.01)]
    if len(failed):
        ex = failed.drop_duplicates(["law", "method"]).head(8)
        ex = ex.assign(expression=ex["expression"].astype(str).map(_code_expr))
        parts.append(
            "Examples of what was selected instead of the law (noise <= 1 %):\n\n"
            + _md(
                ex[
                    [
                        "law",
                        "method",
                        "noise",
                        "seed",
                        "form_err",
                        "extrap_dev",
                        "expression",
                    ]
                ]
            )
        )
    parts.append("![recovery](../../figures/symbolic/recovery.png)")
    return "\n\n".join(parts)


def _block_pareto() -> str:
    pf = _t("pareto")
    if pf is None:
        return MISSING
    pf = pf.assign(expression=pf["expression"].astype(str).map(_code_expr))
    cols = ["method", "complexity", "loss", "score", "selected", "expression"]
    return (
        "Bohr's formula, 1 % noise, 16 levels, seed 11 (`pareto.csv`). For "
        "PySR the losses are recomputed with this module's loss on the same "
        "data so that the three fronts share an axis:\n\n"
        + _md(pf[cols])
        + "\n\n![pareto](../../figures/symbolic/pareto.png)"
    )


def _block_vocabulary() -> str:
    v = _t("vocabulary")
    if v is None:
        return MISSING
    v = v.assign(expression=v["expression"].astype(str).map(_code_expr))
    g = (
        v.groupby(["vocabulary", "method"])
        .agg(
            recovered=("recovered", "sum"),
            runs=("seed", "count"),
            median_extrap_dev=("extrap_dev", "median"),
        )
        .reset_index()
    )
    return (
        "Planck's law, 24 points in x = 0.5-8, 0.1 % noise; the form check "
        "runs over x = 0.05-15 (`vocabulary.csv`):\n\n"
        + _md(g)
        + "\n\nEvery run:\n\n"
        + _md(
            v[
                [
                    "vocabulary",
                    "method",
                    "seed",
                    "recovered",
                    "form_err",
                    "extrap_dev",
                    "expression",
                ]
            ]
        )
        + "\n\n![vocabulary](../../figures/symbolic/vocabulary.png)"
    )


def _block_sindy() -> str:
    conv = _t("sindy_derivative_convergence")
    amp = _t("sindy_noise_amplification")
    rec = _t("sindy_recovery")
    chosen = _j("sindy_chosen")
    if conv is None:
        return MISSING
    parts = [
        "Derivative of sin(2t) on a clean signal vs step "
        "(`sindy_derivative_convergence.csv`). Central differences converge "
        "at the observed order in the last column; Savitzky-Golay at a fixed "
        "0.4 s window stops improving, which is its smoothing bias:\n\n" + _md(conv)
    ]
    if amp is not None:
        a = (
            amp.groupby(["sigma", "method"])
            .agg(rms_err=("rms_err", "mean"), fd_theory=("fd_theory", "first"))
            .reset_index()
        )
        parts.append(
            "Noise in the derivative at dt = 0.01 (`sindy_noise_amplification.csv`). "
            "For central differences the measured error matches "
            "sigma/(sqrt(2) dt):\n\n" + _md(a)
        )
    if chosen:
        c = pd.DataFrame(chosen["chosen"])
        parts.append(
            "Threshold and window chosen on seeds 3/7/19 (`sindy_tune.csv`, "
            "`sindy_chosen.json`):\n\n" + _md(c)
        )
    if rec is not None:
        g = (
            rec.groupby(["system", "method", "noise"])
            .agg(
                support=("support_ok", "mean"),
                coef_rel_err=("coef_rel_err", "median"),
                extra_terms=("n_extra", "mean"),
            )
            .reset_index()
        )
        parts.append(
            "SINDy on the reporting seeds (`sindy_recovery.csv`): fraction with "
            "exactly the right terms, median relative coefficient error, mean "
            "number of spurious terms:\n\n" + _md(g)
        )
    parts.append("![sindy](../../figures/symbolic/sindy.png)")
    return "\n\n".join(parts)


def _block_packages() -> dict[str, str] | None:
    p = _j("packages_checks")
    if p is None:
        return None
    cov = pd.DataFrame(
        [
            {
                "stated sigma": k.split("_")[1],
                "absolute_sigma": k.split("_")[-1],
                "coverage": v,
            }
            for k, v in p["coverage_1sigma"].items()
        ]
    )
    adam = pd.DataFrame(
        [
            {"gradient scale": k, "first step": v}
            for k, v in p["adam_first_step"].items()
        ]
    )
    return {
        "coverage": (
            f"Coverage of the 1-sigma interval on the decay rate b over "
            f"{p['coverage_n_sim']} simulated data sets (20 points of "
            f"a exp(-b x), true sigma 0.05; the nominal value is 0.683):\n\n" + _md(cov)
        ),
        "jac": (
            f"On one data set, the covariance rebuilt by hand as "
            f"(J^T J)^-1 s^2 from `least_squares`'s Jacobian agrees with "
            f"`curve_fit`'s to {p['pcov_vs_hand_rel_diff']:.1e} (relative); "
            f"'lm' and 'trf' agree on the parameters to "
            f"{p['lm_vs_trf_param_rel_diff']:.1e} and on the covariance to "
            f"{p['lm_vs_trf_pcov_rel_diff']:.1e}. The residual difference is "
            f"the finite-difference Jacobian at slightly different points."
        ),
        "grad": (
            f"Differentiating a first derivative that was computed without "
            f"`create_graph=True` raises: `{p['second_derivative_without_create_graph']}`. "
            f"With it, d2/dt2 sin t = -sin t to {p['second_derivative_max_err']:.1e}."
        ),
        "adam": (
            "Adam's first step, lr = 1e-3, for a gradient of 1e-6, 1 and 1e6:\n\n"
            + _md(adam, ".6g")
        ),
        "versions": (
            f"Checked against the installed scipy {p['scipy_version']}, "
            f"torch {p.get('torch_version')}, sympy {p['sympy_version']}, "
            f"pysr {p['pysr_version']}."
        ),
    }


# ---------------------------------------------------------------------------
# the pages
# ---------------------------------------------------------------------------


def _sr_page() -> str:
    from .text import SR_TEMPLATE

    return SR_TEMPLATE.format(
        growth=_block_growth(),
        recovery=_block_recovery(),
        pareto=_block_pareto(),
        vocabulary=_block_vocabulary(),
        sindy=_block_sindy(),
    )


def _packages_page() -> str:
    from .text import PACKAGES_TEMPLATE

    b = _block_packages()
    if b is None:
        b = dict.fromkeys(("coverage", "jac", "grad", "adam", "versions"), MISSING)
    return PACKAGES_TEMPLATE.format(**b)


def render_doc(docs_dir: Path | None = None) -> list[Path]:
    out_dir = docs_dir or (get_settings().root / "docs" / "theory")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (
        ("symbolic_regression.md", _sr_page()),
        ("packages.md", _packages_page()),
    ):
        p = out_dir / name
        p.write_text(text)
        paths.append(p)
    return paths
