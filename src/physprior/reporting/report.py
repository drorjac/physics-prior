"""Generate the headline numbers, docs/RESULTS.md and the README's results
section, all from `results/`.

`physprior report`

The rule this file exists to enforce: **no number in the README, the docs or
the notebooks is typed by hand.** Add a number here, computed from a results
file; never a literal in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json, load_table

# The benchmark tracks, by problem. `relativity/mercury` is deliberately
# absent: it is a single linear solve, not a five-arm benchmark, so it has no
# sweeps to summarise. Its numbers enter through `parameter_recovery`.
TRACKS = {
    "gravity/kepler": "gravity · Kepler's third law (JPL DE441)",
    "relativity/gw150914": "relativity · GW150914 inspiral (LIGO)",
    "quantum/hydrogen": "quantum · hydrogen levels (NIST ASD)",
    "quantum/cmb": "quantum · CMB blackbody (COBE/FIRAS)",
}

ARMS = ["oracle", "physics", "pinn", "sr", "nn"]


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def parameter_recovery() -> list[dict]:
    rows = []

    m = _try(lambda: load_json("relativity/gw150914", "meta"))
    if m:
        pub = m["published_Mc_detector"]
        for r in m["pn_ablation"]:
            if not r.get("converged"):
                rows.append(
                    {
                        "track": "relativity",
                        "quantity": f"chirp mass Mc [{r['label']}]",
                        "published": pub,
                        "recovered": None,
                        "sigma": None,
                        "deviation": "fit pinned to bound (did not converge)",
                    }
                )
                continue
            rows.append(
                {
                    "track": "relativity",
                    "quantity": f"chirp mass Mc [{r['label']}]",
                    "published": pub,
                    "recovered": r["Mc"],
                    "sigma": r["Mc_sigma"],
                    "deviation": f"{r['bias_Msun']:+.2f} Msun ({r['bias_sigma']:+.2f} sigma)",
                }
            )

    m = _try(lambda: load_json("quantum/cmb", "meta"))
    if m and "headline" in m:
        p = m["headline"]["physics"]
        rows.append(
            {
                "track": "quantum",
                "quantity": "CMB temperature T [K]",
                "published": m["published_T_K"],
                "recovered": p["params"]["T"],
                "sigma": p["sigma"].get("T"),
                "deviation": f"{(p['params']['T'] - m['published_T_K']) / m['published_T_K'] * 1e6:+.0f} ppm "
                "(partly by construction - see caveat)",
            }
        )

    m = _try(lambda: load_json("quantum/hydrogen", "meta"))
    if m and "headline" in m:
        p = m["headline"]["physics"]
        rec = p["params"]["R"]
        rows.append(
            {
                "track": "quantum",
                "quantity": "Rydberg R vs NIST ionisation limit [cm^-1]",
                "published": m["ionisation_limit_icm"],
                "recovered": rec,
                "sigma": p["sigma"].get("R"),
                "deviation": f"{(rec - m['ionisation_limit_icm']) / m['ionisation_limit_icm'] * 1e9:+.0f} ppb",
            }
        )
        rows.append(
            {
                "track": "quantum",
                "quantity": "Rydberg R vs Bohr prediction [cm^-1]",
                "published": m["bohr_rydberg_H_icm"],
                "recovered": rec,
                "sigma": p["sigma"].get("R"),
                "deviation": f"{(rec - m['bohr_rydberg_H_icm']) / m['bohr_rydberg_H_icm'] * 1e6:+.2f} ppm "
                "= QED + relativistic",
            }
        )

    m = _try(lambda: load_json("gravity/kepler", "meta"))
    if m and "headline" in m:
        p = m["headline"]["physics"]
        rows.append(
            {
                "track": "gravity",
                "quantity": "GM_sun from Kepler [m^3/s^2]",
                "published": m["published_GM_sun"],
                "recovered": p["params"]["GM"],
                "sigma": p["sigma"].get("GM"),
                "deviation": f"{(p['params']['GM'] - m['published_GM_sun']) / m['published_GM_sun'] * 1e6:+.1f} ppm",
            }
        )

    m = _try(lambda: load_json("relativity/mercury", "meta"))
    if m and "gr" in m:
        g = m["gr"]
        rows.append(
            {
                "track": "relativity",
                "quantity": "GR coefficient alpha (Mercury)",
                "published": 1.0,
                "recovered": g["alpha_GR"],
                "sigma": g["alpha_sigma"],
                "deviation": f"{g['alpha_minus_one']:+.2e}",
            }
        )
        rows.append(
            {
                "track": "relativity",
                "quantity": "perihelion advance [arcsec/century]",
                "published": g["precession_published"],
                "recovered": g["precession_arcsec_cy"],
                "sigma": None,
                "deviation": f"{g['precession_arcsec_cy'] - g['precession_published']:+.3f}",
            }
        )
    return rows


def _ratio(num, den):
    return num / den if den > 0 and np.isfinite(num) else np.inf


def _span(values) -> str:
    """`lo – hi` over the reporting seeds, in the table's own number format."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return ""
    return f"{_fmt_metric(v.min())} – {_fmt_metric(v.max())}"


def extrapolation_summary() -> list[dict]:
    """Median over the reporting seeds, with the per-seed spread beside it.

    A median of three seeds does not show whether the seeds agree. `out/in range` is the per-seed min and max of the same ratio.
    """
    rows = []
    for key in TRACKS:
        ex = _try(lambda k=key: load_table(k, "extrapolation"))
        if ex is None:
            continue
        g = ex.groupby("arm")[["nrmse_in", "nrmse_out"]].median()
        for arm in ARMS:
            if arm not in g.index:
                continue
            a, b = float(g.loc[arm, "nrmse_in"]), float(g.loc[arm, "nrmse_out"])
            per_seed = ex[ex["arm"] == arm]
            ratios = [
                _ratio(o, i)
                for i, o in zip(per_seed["nrmse_in"], per_seed["nrmse_out"])
            ]
            rows.append(
                {
                    "track": key,
                    "arm": arm,
                    "n_seeds": int(per_seed["seed"].nunique()),
                    "nrmse_in": a,
                    "nrmse_out": b,
                    "out/in": _ratio(b, a),
                    "out/in range": _span(ratios),
                }
            )
    return rows


def extrapolation_vs_physics() -> list[dict]:
    """Each arm against `physics` out of range, paired seed by seed.

    `ratio` is arm error / physics error on the same seed and split, so a
    value above 1 means the arm is worse. The verdict is stated only when
    every reporting seed falls on the same side of 1; otherwise it is
    `mixed`. With three seeds a p-value carries little information, so the
    table reports sign agreement instead.
    """
    rows = []
    for key in TRACKS:
        ex = _try(lambda k=key: load_table(k, "extrapolation"))
        if ex is None:
            continue
        wide = ex.pivot_table(index="seed", columns="arm", values="nrmse_out")
        if "physics" not in wide.columns:
            continue
        for arm in ("pinn", "sr", "nn"):
            if arm not in wide.columns:
                continue
            r = (wide[arm] / wide["physics"]).dropna().to_numpy()
            if len(r) == 0:
                continue
            if np.all(r > 1):
                verdict = "physics better on every seed"
            elif np.all(r < 1):
                verdict = f"{arm} better on every seed"
            else:
                verdict = "mixed"
            rows.append(
                {
                    "track": key,
                    "arm": arm,
                    "n_seeds": len(r),
                    "ratio (median)": float(np.median(r)),
                    "ratio range": _span(r),
                    "verdict": verdict,
                }
            )
    return rows


def capability_matrix() -> list[dict]:
    """What each arm can do, read off the results rather than asserted."""
    caps = []
    for arm in ARMS:
        returns_const, returns_law, n_tracks = 0, 0, 0
        for key in TRACKS:
            ex = _try(lambda k=key: load_table(k, "extrapolation"))
            if ex is None or arm not in set(ex.arm):
                continue
            n_tracks += 1
            sub = ex[ex.arm == arm]
            if any(
                c.startswith("param_") and sub[c].notna().any() for c in sub.columns
            ):
                returns_const += 1
            if sub["expression"].notna().any():
                returns_law += 1
        caps.append(
            {
                "arm": arm,
                "returns a physical constant": f"{returns_const}/{n_tracks} tracks",
                "returns a closed form": (
                    f"{returns_law}/{n_tracks} tracks"
                    + (" (law + correction)" if arm == "pinn" else "")
                ),
                "needs the law in advance": arm in ("oracle", "physics", "pinn"),
                "can be wrong in a new way": {
                    "oracle": "no - it is the reference",
                    "physics": "yes - a truncated law gives a confident wrong constant",
                    "pinn": "yes - the network can absorb the physics if w_phys is too small",
                    "sr": "yes - fits the band, misses the law",
                    "nn": "no law to be wrong about; fails outside the range instead",
                }[arm],
            }
        )
    return caps


def budget_crossover() -> list[dict]:
    """The training budget at which the black box first beats the physics arm.
    `None` means it never does, within the budgets swept."""
    out = []
    for key in TRACKS:
        b = _try(lambda k=key: load_table(k, "sweep_budget"))
        if b is None:
            continue
        g = b.groupby(["arm", "n_train"])["nrmse_out"].median().unstack(0)
        if "nn" not in g or "physics" not in g:
            continue
        wins = list(g.index[(g["nn"] < g["physics"]).to_numpy()])
        # Reporting only the SMALLEST winning budget would be misleading: at
        # 3-4 points the comparison is noise (the physics arm has 2 free
        # parameters), and the win does not persist. Report the whole set and
        # whether it still holds at the largest budget.
        out.append(
            {
                "track": key,
                "budgets swept": f"{int(g.index.min())}-{int(g.index.max())}",
                "budgets where nn beats physics": ", ".join(str(int(w)) for w in wins)
                or "none",
                "nn wins at largest budget": bool(
                    len(wins) and wins[-1] == g.index.max()
                ),
                "nn nRMSE at max budget": float(g["nn"].iloc[-1]),
                "physics nRMSE at max budget": float(g["physics"].iloc[-1]),
                "physics advantage": float(g["nn"].iloc[-1] / g["physics"].iloc[-1]),
            }
        )
    return out


# Every case where an arm beats the oracle, and why. A flag with no
# explanation is not a result; leaving one unexplained would mean the setup is
# wrong somewhere.
ORACLE_NOTES = {
    "relativity/gw150914": (
        "in-range: 4-5 training points against 2+ free parameters is an "
        "overfit, as expected. Out-of-range nothing beats it, which is the "
        "real test."
    ),
    "quantum/cmb": (
        "the oracle carries T = 2.72548 K (Fixsen 2009) while the "
        "distributed FIRAS monopole is built on a 2.725 K blackbody, so a "
        "fitted T is closer to THIS dataset by construction."
    ),
    "quantum/hydrogen": (
        "the oracle carries Bohr's R_H, which the data says is 10.8 ppm "
        "low. Beating it is the QED result, not overfitting."
    ),
    "gravity/kepler": (
        "the oracle carries the IAU nominal GM_sun; a fitted GM absorbs the "
        "planet masses and the osculating-vs-mean semi-major axis, both "
        "omitted from P = 2 pi sqrt(a^3/GM)."
    ),
}


def oracle_sanity() -> list[dict]:
    """The oracle is the ceiling. Any arm that beats it INSIDE the training
    range is fitting noise, and any arm that cannot be beaten OUTSIDE it has
    generalised. Both are checked here rather than asserted in prose."""
    rows = []
    for key in TRACKS:
        ex = _try(lambda k=key: load_table(k, "extrapolation"))
        if ex is None or "oracle" not in set(ex.arm):
            continue
        g = ex.groupby("arm")[["nrmse_in", "nrmse_out"]].median()
        o_in, o_out = (
            float(g.loc["oracle", "nrmse_in"]),
            float(g.loc["oracle", "nrmse_out"]),
        )
        beat_in = [a for a in g.index if a != "oracle" and g.loc[a, "nrmse_in"] < o_in]
        beat_out = [
            a for a in g.index if a != "oracle" and g.loc[a, "nrmse_out"] < o_out
        ]
        tr = key
        rows.append(
            {
                "track": tr,
                "oracle nRMSE in": o_in,
                "oracle nRMSE out": o_out,
                "beat oracle in-range": ", ".join(beat_in) or "none",
                "beat oracle out-of-range": ", ".join(beat_out) or "none",
                "explanation": (
                    ORACLE_NOTES.get(tr, "") if (beat_in or beat_out) else "-"
                ),
            }
        )
    return rows


def headline() -> dict:
    return {
        "parameter_recovery": parameter_recovery(),
        "extrapolation_summary": extrapolation_summary(),
        "extrapolation_vs_physics": extrapolation_vs_physics(),
        "capability_matrix": capability_matrix(),
        "budget_crossover": budget_crossover(),
        "oracle_sanity": oracle_sanity(),
    }


def render(h: dict) -> dict[Path, str]:
    """Every file `physprior report` owns, and the text it should hold."""
    settings = get_settings()
    out = {
        settings.results_dir / "headline.json": json.dumps(h, indent=2, default=str),
        settings.root / "docs" / "RESULTS.md": _results_md(h),
    }
    readme = settings.root / "README.md"
    text = _readme_with_results(readme, h)
    if text is not None:
        out[readme] = text
    return out


def build() -> dict:
    h = headline()
    for path, text in render(h).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return h


def check() -> list[Path]:
    """The generated files that no longer match `results/`.

    Empty means the README, docs/RESULTS.md and headline.json are exactly
    what `physprior report` would write now: nobody edited a generated table
    by hand, and nobody changed `results/` without re-running the report.
    """
    stale = []
    for path, text in render(headline()).items():
        if not path.is_file() or path.read_text() != text:
            stale.append(path)
    return stale


# Physical constants are compared digit by digit against their published
# values: 6 significant figures turns 109678.7774 into "109679" and loses the
# entire point of track A. Every other column is an error, a ratio or a
# sigma, and three significant figures is all a reader can use of those.
_PRECISE_COLUMNS = {"published", "recovered"}


def _fmt_precise(v) -> str:
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e6 or a < 1e-4:
        return f"{v:.6e}"
    return f"{v:.10g}" if a >= 1e3 else f"{v:.6g}"


def _fmt_metric(v) -> str:
    """Three significant figures; scientific notation only outside 1e-3..1e5."""
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e5:
        return f"{v:.2e}"
    return f"{float(f'{v:.3g}'):,.{max(0, 2 - int(np.floor(np.log10(a))))}f}"


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(not yet computed)_\n"
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(_fmt_precise if c in _PRECISE_COLUMNS else _fmt_metric)
    df = df.fillna("")
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)
    )
    return f"{head}\n{sep}\n{body}\n"


def _results_md(h: dict) -> str:
    parts = [
        "# physprior — results\n",
        "_Generated by `physprior report` from `results/`. Do not edit by hand._\n",
        "\n## Physical constants recovered from real data\n",
        _md_table(h["parameter_recovery"]),
        "\n## Extrapolation: error inside vs outside the training range\n",
        "`out/in` is the factor by which the error grows when the arm is "
        "asked to leave the range it was fitted on.\n\n",
        _md_table(h["extrapolation_summary"]),
        "\n### Against the fitted law, seed by seed\n",
        "Each arm's out-of-range error divided by `physics`'s on the same "
        "seed and split; above 1 means the arm is worse. A verdict is given "
        "only when every reporting seed agrees.\n\n",
        _md_table(h["extrapolation_vs_physics"]),
        "\n## Data budget: when does the black box catch up?\n",
        _md_table(h["budget_crossover"]),
        "\n## What each arm can do\n",
        _md_table(h["capability_matrix"]),
        "\n## Sanity: is the oracle ever beaten?\n",
        "The oracle is the published law with published constants. An arm "
        "that beats it INSIDE the training range is fitting noise; an arm "
        "that beats it OUTSIDE is a genuine improvement on the textbook "
        "and should be treated with suspicion until explained.\n\n",
        _md_table(h["oracle_sanity"]),
    ]
    return "".join(parts)


START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _readme_with_results(rd: Path, h: dict) -> str | None:
    """The README with its generated block replaced, or None if it has none."""
    if not rd.exists():
        return None
    text = rd.read_text()
    if START not in text or END not in text:
        return None
    body = (
        "\n"
        + _md_table(h["parameter_recovery"])
        + "\nExtrapolation — error outside the training range relative to inside:\n\n"
        + _md_table(h["extrapolation_summary"])
        + "\n"
    )
    a, b = text.index(START) + len(START), text.index(END)
    return text[:a] + body + text[b:]


if __name__ == "__main__":
    out = build()
    print(json.dumps({k: len(v) for k, v in out.items()}, indent=2))
    print("wrote results/headline.json, docs/RESULTS.md and the README section")
