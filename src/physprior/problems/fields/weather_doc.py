"""docs/fields/weather.md, generated from results/fields/weather/.

Every number on the page is read from a results file here; the prose around
the tables describes method only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json, load_table

from . import weather as W
from . import weather_sim as S
from .weather_plots import LABEL, ORDER

DOC = "fields/weather.md"


def _fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}"
    return str(v)


def _table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def _load(tr: str, name: str) -> pd.DataFrame | None:
    try:
        return load_table(tr, name)
    except FileNotFoundError:
        return None


def _fig(name: str, alt: str) -> str:
    return f"![{alt}](../../figures/fields/{name}.png)"


# ---------------------------------------------------------------------------


def data_table(cases) -> pd.DataFrame:
    rows = []
    for c in cases:
        m = load_json(W.track(c), "meta")
        p = m["provenance"]
        rows.append(
            {
                "case": c,
                "stations": m["n_stations"],
                "height km": f"{m['z_km_range'][0]:.3f} to {m['z_km_range'][1]:.3f}",
                "T degC": f"{m['temp_c_range'][0]:.1f} to {m['temp_c_range'][1]:.1f}",
                "hourly files": p["hourly_files"],
                "MB": p["hourly_bytes"] / 1e6,
            }
        )
    return pd.DataFrame(rows)


def gamma_table(cases) -> pd.DataFrame:
    rows = []
    for c in cases:
        h = load_json(W.track(c), "meta")["headline"]
        ph, uk = h["physics"], h["uk"]
        rows.append(
            {
                "case": c,
                "physics Gamma": ph["params"]["Gamma"],
                "± (OLS)": ph["sigma"]["Gamma"],
                "pull vs -6.5 (OLS)": h["gamma_minus_published_sigma"],
                "PINN Gamma": h["pinn"]["params"]["Gamma"],
                "PINN mean dT/dz": h["pinn"]["dTdz"],
                "UK Gamma": uk["params"]["Gamma"],
                "± (GLS)": uk["sigma"]["Gamma"],
                "pull vs -6.5 (GLS)": h["uk_gamma_minus_published_sigma"],
            }
        )
    return pd.DataFrame(rows)


def _with_kriging(tr: str, split: str) -> pd.DataFrame | None:
    name = "extrapolation" if split == "elevation" else "blocks"
    ex = _load(tr, name)
    kr = _load(tr, "kriging")
    if ex is None:
        return None
    if kr is not None:
        kr = kr[kr.sweep == split].copy()
        kr["seed"] = np.nan
        ex = pd.concat([ex, kr], ignore_index=True)
    return ex


def elevation_table(tr: str) -> pd.DataFrame | None:
    df = _with_kriging(tr, "elevation")
    if df is None:
        return None
    rows = []
    for a in [a for a in ORDER if (df.arm == a).any()]:
        g = df[df.arm == a]
        seeds = g.dropna(subset=["seed"]) if g.seed.notna().any() else g
        rows.append(
            {
                "arm": LABEL[a],
                "nRMSE in": float(g.nrmse_in.median()),
                "nRMSE out (median)": float(g.nrmse_out.median()),
                "per seed": ", ".join(f"{v:.3f}" for v in seeds.nrmse_out),
                "RMSE out K": float(g.rmse_out.median()),
                "mean dT/dz": float(g.dTdz.median()),
            }
        )
    return pd.DataFrame(rows)


def block_table(tr: str) -> pd.DataFrame | None:
    df = _with_kriging(tr, "block")
    if df is None:
        return None
    rows = []
    for a in [a for a in ORDER if (df.arm == a).any()]:
        g = df[df.arm == a]
        row: dict[str, object] = {"arm": LABEL[a]}
        for b in sorted(g.block.unique()):
            row[f"block {int(b)}"] = float(g[g.block == b].nrmse_out.median())
        row["RMS over blocks"] = float(np.sqrt(np.mean(g.nrmse_out**2)))
        rows.append(row)
    return pd.DataFrame(rows)


def kriging_hyper_table(cases) -> pd.DataFrame:
    rows = []
    for c in cases:
        kr = _load(W.track(c), "kriging")
        if kr is None:
            continue
        for _, r in kr[kr.sweep == "elevation"].iterrows():
            rows.append(
                {
                    "case": c,
                    "arm": LABEL[r.arm],
                    "l_h km": r.hyper_l_h_km,
                    "l_z km": r.hyper_l_z_km,
                    "s_f K": r.hyper_s_f,
                    "s_n K": r.hyper_s_n,
                    "converged": bool(r.converged),
                    "starts agreeing": f"{int(r.starts_agreeing)}/3",
                }
            )
    return pd.DataFrame(rows)


def sim_table() -> pd.DataFrame | None:
    rows = []
    meta = load_json(S.TRACK_ROOT, "meta")
    for case in S.CASES:
        df = _load(S.track(case), "extrapolation")
        if df is None:
            return None
        for a in [a for a in ORDER if (df.arm == a).any()]:
            g = df[df.arm == a]
            rows.append(
                {
                    "case": case,
                    "arm": LABEL[a],
                    "nRMSE out": float(g.nrmse_out.median()),
                    "mean dT/dz": float(g.dTdz.median()),
                    "free-air Gamma": S.TRUTH["Gamma"],
                    "best linear Gamma": meta["gamma_best_linear"][case],
                }
            )
    return pd.DataFrame(rows)


def coverage_table() -> pd.DataFrame | None:
    df = _load(S.TRACK_ROOT, "coverage_summary")
    if df is None:
        return None
    df = df.copy()
    df["arm"] = df.arm.map(LABEL)
    return df[
        [
            "case",
            "arm",
            "draws",
            "gamma_ref",
            "gamma_mean",
            "gamma_sd_over_draws",
            "sigma_quoted_median",
            "coverage_1sigma",
            "pull_sd",
        ]
    ]


def sweep_table(tr: str, name: str, key: str) -> pd.DataFrame | None:
    df = _load(tr, name)
    if df is None:
        return None
    p = df.pivot_table(index="arm", columns=key, values="nrmse_out", aggfunc="median")
    p = p.reindex([a for a in ORDER if a in p.index])
    p.index = [LABEL[a] for a in p.index]
    p.columns = [f"{key} = {c:g}" for c in p.columns]
    return p.reset_index(names="arm")


def steps_table(cases) -> pd.DataFrame:
    rows = []
    for c in cases:
        st = _load(W.track(c), "lapse_step_convergence")
        if st is None:
            continue
        for a, g in st.groupby("arm", sort=False):
            row = {"case": c, "arm": LABEL[a]}
            for _, r in g.iterrows():
                row[f"h = {r.h_km:g} km"] = r.dTdz
            rows.append(row)
    return pd.DataFrame(rows)


def best_out(tr: str) -> str:
    df = elevation_table(tr)
    if df is None or df.empty:
        return ""
    skip = {LABEL["oracle"], LABEL["pinn_static"]}
    df = df[~df.arm.isin(skip)]
    r = df.loc[df["nRMSE out (median)"].idxmin()]
    return (
        f"Lowest median held-out nRMSE among the arms and kriging (oracle and "
        f"the fixed-start diagnostic excluded): {r.arm} "
        f"({r['nRMSE out (median)']:.3f})."
    )


# ---------------------------------------------------------------------------


def render_doc(cases=tuple(W.CASES)) -> Path:
    s = get_settings()
    parts = [
        """# fields/weather: near-surface temperature over the Alps

Code: [`src/physprior/problems/fields`](../../src/physprior/problems/fields)
· Results: [`results/fields/weather`](../../results/fields/weather)
· Tutorial: T10

*Generated by `physprior.problems.fields.weather_doc.render_doc()` from the
results files. Do not edit by hand.*

## Question

A monthly temperature field sampled at weather stations, with a physical law
for its large-scale structure: temperature falls linearly with height,

    T = T0 + a (lon - lon0) + b (lat - lat0) + Gamma z,     z in km.

The published reference for Gamma is the ICAO / ISO 2533 standard atmosphere,
-6.5 K/km (ICAO Doc 7488/3, 1993). It is a free-atmosphere value. Lapse
rates measured along mountain slopes near the ground are known to differ
from it and to vary with season (Rolland 2003; Kirchner et al. 2013), so the
fitted Gamma is compared with -6.5 as a reference, not as a truth. T0, a and
b have no published value. The `oracle` arm therefore holds Gamma at -6.5
and fits T0, a and b on the training stations; it tests the standard
lapse rate, and it is not a ceiling.

The `pinn` arm is the project's frozen PINN with its constants scaled so
Adam can reach them: T0 starts at the mean training temperature and Gamma is
optimised in log space from -1 K/km. This was chosen on the tuning seeds by
one criterion, that the PINN's constants reach the least-squares constants
on the same training stations. The frozen PINN from fixed starts
(T0 = 10 degC, Gamma = 0) is reported as `PINN, fixed start` on the
elevation split: an additive constant moves by about one learning rate per
Adam step, and those constants do not get there in 4000 epochs.

The arms see x = (z, lon, lat). Two held-out splits:

* **elevation**: train on the lowest 75% of stations by height, test on the
  highest 25%. The black box has never seen a mountain top.
* **longitude blocks**: four longitude quartiles, each held out in turn.

Kriging (Gaussian-process regression, ML-II on the training stations) is run
on the same splits: ordinary kriging with a constant mean, and universal
kriging with the law as the mean. They are reported beside the arms.

## Data

NOAA Integrated Surface Database, ISD-Lite hourly files, box lon 5.8 to
13.5 E, lat 45.5 to 48.0 N. The snapshot rule was fixed before any fit: the
mean over the month of the 12 UTC air temperature, for stations with a 12 UTC
reading on at least 80% of days, co-located duplicate records merged.
July 2023 is a mixed summer boundary layer; January 2023 is the harder case,
with valley cold pools in which the linear law is known to be wrong.
""",
        _table(data_table(cases)),
        "",
        _fig("weather_data", "stations and temperature against height"),
        """
## Recovered lapse rate (all stations)

`physics` is ordinary least squares, and its error bar assumes independent
residuals. Universal kriging estimates the same four constants by
generalised least squares under the fitted spatial covariance, so its error
bar allows for stations that share weather. The pull is
(Gamma - (-6.5)) / sigma.
""",
        _table(gamma_table(cases)),
        """
## Elevation extrapolation

Median over the reporting seeds 11, 23, 42. nRMSE is RMSE divided by the
standard deviation of all station temperatures. Kriging is deterministic and
has one value. `mean dT/dz` is the derivative of each fitted function with
respect to height, averaged over all stations (central differences; its
step convergence is tabulated below).
""",
    ]
    for c in cases:
        t = elevation_table(W.track(c))
        if t is not None:
            parts += [
                f"### {c}\n",
                _table(t),
                "",
                best_out(W.track(c)),
                "",
                _fig(f"weather_extrapolation_{c}", f"extrapolation, {c}"),
                "",
                _fig(f"weather_maps_{c}", f"maps, {c}"),
                "",
            ]
    parts += [
        _fig("weather_nrmse", "held-out error per arm"),
        "\n## Longitude blocks\n",
        "Held-out nRMSE per block (block 0 is the westernmost), median over "
        "seeds, and the RMS over the four blocks.\n",
    ]
    for c in cases:
        t = block_table(W.track(c))
        if t is not None:
            parts += [f"### {c}\n", _table(t), ""]
    parts += [
        "\n## Kriging hyperparameters (elevation split)\n",
        "A length scale within 1% (in log) of its bound is marked not "
        "converged. Three fixed starting points; the count that reach the "
        "best likelihood is given.\n",
        _table(kriging_hyper_table(cases)),
        "",
        _fig("weather_gamma", "lapse rate implied by each arm"),
    ]
    sim = sim_table()
    if sim is not None:
        sm = load_json(S.TRACK_ROOT, "meta")
        parts += [
            """
## Simulated control

The same station positions; a field with known constants
(T0, a, b, Gamma) = ({T0}, {a}, {b}, {G}), a GP residual (s_f = {sf} K,
l_h = {lh} km, l_z = {lz} km), {nz} K independent noise, and in the
`inversion` case a cold pool of {C} K at sea level fading linearly to zero
at {zt} km. With the pool, the best linear Gamma at these stations is not
the free-air Gamma; both are listed. One realisation per reporting seed,
elevation split. `mean dT/dz` is taken over all stations of a model fitted
on the lowest 75%; in the `inversion` case most of those training stations
lie inside the pool, where the true slope is the free-air Gamma plus the
pool's gradient.
""".format(
                T0=S.TRUTH["T0"],
                a=S.TRUTH["a"],
                b=S.TRUTH["b"],
                G=S.TRUTH["Gamma"],
                sf=S.RESIDUAL["s_f"],
                lh=S.RESIDUAL["l_h_km"],
                lz=S.RESIDUAL["l_z_km"],
                nz=S.NOISE_K,
                C=S.INVERSION["C"],
                zt=S.INVERSION["z_top_km"],
            ),
            _table(sim),
            f"\n### Are the error bars calibrated?\n\nOver {sm['coverage'][0]['draws']} "
            "simulated draws, fitting all stations: the fraction of draws in "
            "which the quoted 1-sigma interval contains the reference Gamma "
            "(0.683 if the error bar is right), and the standard deviation of "
            "the pulls (1 if right). The reference is the best linear Gamma of "
            "the noise-free field.\n",
            _table(coverage_table()),
            "",
            _fig("weather_sim_coverage", "pull distributions"),
        ]
    for name, key, title in (
        ("sweep_budget", "n_train", "Budget sweep"),
        ("sweep_noise", "noise_frac", "Noise sweep"),
    ):
        t = sweep_table(W.track(W.SWEEP_CASE), name, key)
        if t is not None:
            parts += [
                f"\n## {title} ({W.SWEEP_CASE}, random splits)\n",
                "Median held-out nRMSE. Symbolic regression is left out of the "
                "sweeps for compute; it runs on both splits above.\n",
                _table(t),
            ]
    parts += [
        "\n## Derivative convergence\n",
        "Mean dT/dz by central differences with step h, for models fitted on "
        "the elevation training split. The value must stop moving as h "
        "shrinks.\n",
        _table(steps_table(cases), nd=4),
        """
## Limitations

* One year, two months, one hour of day. Other hours (in particular early
  morning in winter, when inversions are strongest) are not studied.
* Station heights come from the ISD station list and are not checked
  against a terrain model.
* The law has no land-use, slope-aspect or distance-to-lake terms; the
  universal-kriging residual absorbs whatever of these is spatially smooth.
* The simulated cold pool is a function of absolute height. Real cold pools
  sit on each valley floor, at different heights.
""",
    ]
    path = s.root / "docs" / DOC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts).rstrip() + "\n")
    return path
