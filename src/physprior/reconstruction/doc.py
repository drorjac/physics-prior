"""docs/reconstruction/README.md, generated from results/reconstruction/.

Every number on the page is read from a results file; the surrounding prose
describes method only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json, load_table

from . import fields as F

TRACK = "reconstruction"
ORDER = ["physics", "pinn", "pigp", "gp", "interp", "nn"]


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not np.isfinite(x):
            return "–"
        return f"{x:.{nd}g}"
    return str(v)


def _table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def _fig(name: str, alt: str) -> str:
    return f"![{alt}](../../figures/{TRACK}/{name}.png)"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    key = {m: i for i, m in enumerate(ORDER)}
    return df.assign(_k=df.method.map(key)).sort_values(["_k"]).drop(columns="_k")


# ---------------------------------------------------------------------------


def samples_table(ns: pd.DataFrame, metric: str, target: float) -> pd.DataFrame:
    s = ns[(ns.metric == metric) & (ns.target == target)]
    rows = []
    for case in F.CASES:
        for m in ORDER:
            g = s[(s.case == case) & (s.method == m)].set_index("d")
            if g.empty:
                continue
            row = {"case": case, "method": m}
            for d in (1, 2, 3):
                if d not in g.index:
                    row[f"{d}-D"] = "–"
                    continue
                r = g.loc[d]
                v = f"{r.N_star:.0f}"
                if r.censored:
                    v = f"> {r.N_max:.0f}"
                elif r.at_first_grid_point:
                    v = f"<= {r.N_star:.0f}"
                row[f"{d}-D"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    unk_rows = [
        {"case": c, "method": "unknowns P"}
        | {f"{d}-D": str(F.Geometry(c, d).n_unknowns()) for d in (1, 2, 3)}
        for c in F.CASES
    ]
    return pd.concat([df, pd.DataFrame(unk_rows)], ignore_index=True)


def per_d_table(sw: pd.DataFrame, case: str, col: str, methods) -> pd.DataFrame:
    """Median `col` by N for each dimension and method (wide over N)."""
    out = []
    for d in (1, 2, 3):
        g = sw[(sw.case == case) & (sw.d == d) & sw.method.isin(methods)]
        if g.empty:
            continue
        p = g.groupby(["method", "N"])[col].median().unstack("N")
        p = p.reindex([m for m in ORDER if m in p.index])
        p.insert(0, "method", p.index)
        p.insert(0, "d", d)
        p.columns = [str(c) for c in p.columns]
        out.append(p.reset_index(drop=True))
    return out


def growth_table(gr: pd.DataFrame) -> pd.DataFrame:
    g = gr[(gr.metric == "nrmse") & (gr.target == 0.1)]
    g = _sorted(g)[
        [
            "case",
            "method",
            "decades_per_dim",
            "ratio_3d_1d",
            "any_censored",
            "any_at_first_grid_point",
        ]
    ]
    return g.sort_values("case", kind="stable")


def verdict_text(v: dict) -> str:
    lines = []
    for case in F.CASES:
        if case not in v:
            continue
        c = v[case]
        lines.append(f"`{case}`:\n")
        lines.append(
            "| criterion | measured | holds |\n|---|---|---|\n"
            f"| 1. best non-physics N*(3-D)/N*(1-D) > 3 | "
            f"{_fmt(c['black_box_ratio_3d_1d'])}"
            f"{' (lower bound)' if c['black_box_3d_censored'] else ''} | "
            f"{_fmt(c['criterion_1_black_box_grows'])} |\n"
            f"| 2. physics N*(3-D) <= 3 x N*(1-D) P(3)/P(1) | "
            f"{_fmt(c['physics_N_star']['3'])} vs "
            f"{_fmt(3 * c['physics_scaled_prediction_3d'])} | "
            f"{_fmt(c['criterion_2_physics_tracks_unknowns'])} |\n"
            f"| 3. gap N*(best non-physics)/N*(physics) grows 1-D -> 3-D | "
            f"{_fmt(c['gap_1d'])} -> {_fmt(c['gap_3d'])} | "
            f"{_fmt(c['criterion_3_gap_widens'])} |\n"
            f"| all ratios defined (no censored 1-D or physics value) | | "
            f"{_fmt(c.get('determinate', True))} |\n"
            f"| hypothesis supported (all three) | | {_fmt(c['supported'])} |\n"
        )
        best = ", ".join(f"{d}-D: {m}" for d, m in c["best_black_box_by_d"].items())
        lines.append(f"Best non-physics method by dimension: {best}.\n")
    return "\n".join(lines)


def mismatch_table(mm: pd.DataFrame) -> pd.DataFrame:
    p = mm.groupby(["kind", "method", "N"]).nrmse.median().unstack("N")
    p = p.reset_index()
    p.columns = [str(c) for c in p.columns]
    key = {m: i for i, m in enumerate(ORDER)}
    return p.assign(_k=p.method.map(key)).sort_values(["kind", "_k"]).drop(columns="_k")


def location_table(sw: pd.DataFrame) -> list[pd.DataFrame]:
    s = sw[sw.method.isin(["physics", "pinn"])]
    out = []
    for case in F.CASES:
        g = s[s.case == case]
        if g.empty:
            continue
        p = g.groupby(["d", "method", "N"]).loc_err.median().unstack("N").reset_index()
        p.columns = [str(c) for c in p.columns]
        out.append((case, p))
    return out


def timing_table(sw: pd.DataFrame) -> pd.DataFrame:
    g = sw.groupby(["d", "method"]).seconds.median().unstack("method")
    g = g[[m for m in ORDER if m in g.columns]].reset_index()
    return g


def noise_table(nz: pd.DataFrame) -> pd.DataFrame:
    p = (
        nz.groupby(["d", "method", "noise"])
        .nrmse.median()
        .unstack("noise")
        .reset_index()
    )
    p.columns = [str(c) for c in p.columns]
    return p


def render_doc(path: Path | None = None) -> Path:
    s = get_settings()
    path = path or (s.root / "docs" / TRACK / "README.md")
    meta = load_json(TRACK, "meta")
    v = load_json(TRACK, "verdict")
    tuned = load_json(TRACK, "tuning")
    sw = load_table(TRACK, "sweep")
    ns = load_table(TRACK, "samples_needed")
    gr = load_table(TRACK, "growth")
    mm = load_table(TRACK, "mismatch")
    nz = load_table(TRACK, "noise")
    conv_fd = load_table(TRACK, "convergence_fd_series")
    conv_mms = load_table(TRACK, "convergence_fd_mms")
    conv_s = load_table(TRACK, "convergence_series")
    conv_free = load_table(TRACK, "convergence_free")

    L: list[str] = []
    a = L.append
    a(
        "# Field reconstruction from sparse sensors: does the physics prior scale with dimension?\n"
    )
    a(
        "Generated by `physprior.reconstruction.doc.render_doc()` from "
        "`results/reconstruction/`. Do not edit by hand. The hypothesis and its "
        "refutation criteria were written before the study ran: "
        "[HYPOTHESIS.md](HYPOTHESIS.md).\n"
    )
    if meta.get("quick"):
        a(
            "> This page was rendered from a `quick` run (one seed, tiny grids). "
            "Its numbers are smoke-test values, not results.\n"
        )
    a("## Setup\n")
    a(
        f"- Fields: Poisson's equation with {meta['n_sources']} Gaussian sources "
        f"of width {meta['source_width']} (box side = 1), in 1-D, 2-D and 3-D.\n"
        "  - `box`: steady heat with Dirichlet walls held at an unknown linear "
        "temperature a0 + a.x. Unknowns P = K(d+1) + (d+1). Truth: the exact "
        f"sine-series solution, kmax per axis {meta['kmax_truth']}.\n"
        "  - `free`: potential of mixed-sign charges in free space (closed-form "
        "Green's functions), plus an unknown offset. P = K(d+1) + 1. Sensors in "
        "the unit cube; the field is scored inside it and, as extrapolation, on "
        "the shell out to [-0.5, 1.5]^d.\n"
        f"- Sensors: N uniform random points, Gaussian noise of {meta['noise']} "
        "field standard deviations. Designs are nested in N.\n"
        f"- Seeds: tuning {meta['tune_seeds']}, reporting {meta['seeds']}. Each "
        "seed draws a new field and a new design.\n"
        "- Error: nRMSE = RMSE / std(truth in the sensor region), on a dense "
        "grid. `box`: whole domain; the in/out split is the convex hull of the "
        "sensors. `free`: inside the sensor cube; `out` is the shell.\n"
        f"- The physics fit is run up to N = {meta['physics_n_max']}; beyond "
        "that it is at the noise floor.\n"
    )
    a(
        "Methods: `physics` (Green's function in the loop; matching-pursuit start "
        "on a candidate grid, then bounded least squares from that and from random "
        "starts), `pinn` (MLP + PDE residual with trainable sources; run at two "
        "N per dimension, `box` only), `pigp` (GP on the physics residual), `gp` "
        "(RBF, ML-II), `interp` (thin-plate RBF, smoothing tuned), `nn` (MLP, "
        "tuned).\n"
    )
    a(
        f"Tuned choices (tuning seeds only): MLP per dimension "
        f"`{tuned['nn']}`, RBF smoothing `{tuned['interp']}`, "
        f"PINN w_pde fixed at {tuned['pinn_w_pde']} (not tuned).\n"
    )

    a("## Headline: sensors needed to reach nRMSE 0.1\n")
    a(_fig("samples_needed", "sensors needed vs dimension"))
    a("")
    a(_table(samples_table(ns, "nrmse", 0.1)))
    a(
        "\n`> N`: never reached within the budget (lower bound). `<= N`: already "
        "reached at the smallest N tried.\n"
    )
    a("Stricter target, nRMSE 0.03:\n")
    a(_table(samples_table(ns, "nrmse", 0.03)))
    a(
        "\nGrowth of N* with dimension (target 0.1; slope of log10 N* on d; "
        "censored or first-grid-point values make the slope a bound):\n"
    )
    a(_table(growth_table(gr)))
    a("\n## Verdict on the pre-registered hypothesis\n")
    a(verdict_text(v))

    a("## Error against number of sensors\n")
    for case in F.CASES:
        a(_fig(f"error_vs_n_{case}", f"error vs N, {case}"))
        a("")
        for t in per_d_table(sw, case, "nrmse", ORDER):
            a(_table(t))
            a("")
    a("## Extrapolation\n")
    a(
        "Error outside the region the sensors cover: outside their convex hull "
        "(`box`), on the shell around the sensor cube (`free`).\n"
    )
    a(_fig("extrapolation_free", "extrapolation, free-space potential"))
    a("")
    for t in per_d_table(sw, "free", "nrmse_out", ORDER):
        a(_table(t))
        a("")
    a(_fig("extrapolation_box", "outside the sensor hull, box"))
    a("")
    a("Sensors needed for nRMSE 0.1 outside the sensors:\n")
    a(_table(samples_table(ns, "nrmse_out", 0.1)))
    a("")

    a("## Where are the sources?\n")
    a(
        "Median distance between true and recovered source positions (optimal "
        "matching; box side = 1).\n"
    )
    for case, t in location_table(sw):
        a(f"`{case}`:\n")
        a(_table(t))
        a("")
    n_bound = int(sw[(sw.method == "physics")].at_bound.sum())
    n_phys = int((sw.method == "physics").sum())
    a(
        f"Physics fits with a source position on its search bound "
        f"(not converged, flagged): {n_bound} of {n_phys}.\n"
    )

    a("## Model mismatch (2-D plate)\n")
    a(
        "The physics model is given three sources and Dirichlet walls. The truth "
        "has either a fourth, weaker source or an insulated wall. `pigp` adds a GP "
        "on the physics residual; the dashed line is the physics fit on the clean "
        "field.\n"
    )
    a(_fig("mismatch", "model mismatch"))
    a("")
    a(_table(mismatch_table(mm)))
    a("")

    a("## Noise\n")
    a(_fig("noise", "noise sweep"))
    a("")
    a(_table(noise_table(nz)))
    a("")

    a("## Compute\n")
    a("Median seconds per fit (one thread per worker, shared machine):\n")
    a(_table(timing_table(sw)))
    a("")
    t = meta.get("timing_seconds", {})
    a(
        "Wall time of the run, seconds: "
        + ", ".join(f"{k} {_fmt(x)}" for k, x in t.items())
        + ".\n"
    )

    a("## Numerical checks (convergence studies)\n")
    a(_fig("convergence", "convergence"))
    a("")
    fit_rows = conv_s[conv_s.is_fit_kmax.astype(bool)][["d", "kmax", "rel_max_err"]]
    a("Series truncation at the kmax the physics fit uses:\n")
    a(_table(fit_rows))
    a("\nFD solver against the series (observed order should approach 2):\n")
    a(_table(conv_fd))
    a("\nFD solver against manufactured solutions:\n")
    a(_table(conv_mms))
    a("\nClosed-form free-space potentials, central-difference residual:\n")
    a(_table(conv_free))
    a("")

    a("## Examples\n")
    a(_fig("examples", "example reconstructions"))
    a("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path
