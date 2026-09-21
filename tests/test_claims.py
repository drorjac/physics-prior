"""The narrative must not drift from the results.

The results TABLES in the README and docs/RESULTS.md are generated from
`results/` by `project0.report`, so they cannot drift. The handful of numbers
quoted in the prose -- "+9 Msun", "10.8 ppm", "1.13 +- 0.002" -- are written
by hand, and this file is what stops them going stale: each one is recomputed
from `results/` and checked to still appear in the README.

If one of these fails after a re-run, the number in the README is wrong and
the README is what must change.
"""

from __future__ import annotations

import json

import pytest

from physprior.config import get_settings
from physprior.io import load_json


@pytest.fixture(scope="module")
def readme() -> str:
    return (get_settings().root / "README.md").read_text()


@pytest.fixture(scope="module")
def headline() -> dict:
    p = get_settings().results_dir / "headline.json"
    if not p.exists():
        pytest.skip("results/headline.json missing -- run `make run report`")
    return json.loads(p.read_text())


def _meta(track):
    try:
        return load_json(track, "meta")
    except FileNotFoundError:
        pytest.skip(f"{track} not run yet")


def test_newtonian_chirp_mass_bias(readme):
    """README: the Newtonian law is biased by about +9 Msun."""
    rows = {r["pn_order"]: r for r in _meta("relativity/gw150914")["pn_ablation"]}
    newt = rows[0]
    assert f"+{newt['bias_Msun']:.0f} M" in readme, (
        f"README claims a bias the results do not show: {newt['bias_Msun']:+.2f}"
    )


def test_qed_gap_ppm(readme):
    """README: the fitted Rydberg sits 10.8 ppm above Bohr."""
    ppm = _meta("quantum/hydrogen")["limit_minus_bohr_ppm"]
    assert f"{ppm:.1f} ppm" in readme, f"QED gap drifted to {ppm:.2f} ppm"


def test_false_gr_violation(readme):
    """README: the 4th-order/3-hour run gives alpha = 1.13 +- 0.002."""
    rows = {
        (r["step"], r["fd_order"]): r
        for r in _meta("relativity/mercury")["gr_convergence"]
        if "alpha_GR" in r
    }
    bad = rows[("180m", 4)]
    assert f"{bad['alpha_GR']:.2f}" in readme, (
        f"the coarse alpha drifted to {bad['alpha_GR']:.4f}"
    )
    n_sigma = abs(bad["alpha_GR"] - 1.0) / bad["alpha_sigma"]
    assert f"{n_sigma:.0f} formal sigma" in readme or f"{n_sigma:.0f}σ" in readme, (
        f"the false-violation significance drifted to {n_sigma:.1f} sigma"
    )


def test_converged_gr_agrees_with_einstein(readme):
    """README: with a 6th-order stencil alpha agrees to one part in 10^4."""
    gr = _meta("relativity/mercury")["gr"]
    assert gr["fd_order"] == 6, "the reported GR fit is not the converged one"
    assert abs(gr["alpha_GR"] - 1.0) < 1e-3, (
        f"alpha = {gr['alpha_GR']:.6f} no longer agrees with Einstein"
    )
    assert "10⁴" in readme or "10^4" in readme


def test_physics_weight_table(readme):
    """README: the w_phys table's 'error at w=0' column."""
    import pandas as pd

    for track, param, pub_key in [
        ("gravity/kepler", "GM", "published_GM_sun"),
        ("quantum/cmb", "T", "published_T_K"),
    ]:
        meta = _meta(track)
        d = pd.read_csv(get_settings().results_dir / track / "sweep_physics_weight.csv")
        g = d.groupby("w_phys").median(numeric_only=True)
        pub = meta[pub_key]
        err0 = abs(g.loc[0.0, f"param_{param}"] - pub) / abs(pub) * 100
        assert f"{err0:.3g}".rstrip("0").rstrip(".") in readme.replace("**", ""), (
            f"{track}: w_phys=0 error drifted to {err0:.3f}%"
        )


def test_extrapolation_gap_is_orders_of_magnitude(headline):
    """The claim that the gap is 'not a factor of two' must keep being true
    on the tracks where the parameter is identifiable (Q, A, R)."""
    by = {}
    for r in headline["extrapolation_summary"]:
        by.setdefault(r["track"], {})[r["arm"]] = r["nrmse_out"]
    for tr in ("quantum/cmb", "quantum/hydrogen", "gravity/kepler"):
        if tr not in by:
            continue
        assert by[tr]["nn"] / by[tr]["physics"] > 100, (
            f"track {tr}: nn/physics out-of-range ratio is only "
            f"{by[tr]['nn'] / by[tr]['physics']:.1f}"
        )


def test_oracle_is_not_beaten_out_of_range_without_explanation(headline):
    for row in headline["oracle_sanity"]:
        if row["beat oracle out-of-range"] != "none":
            assert row["explanation"] not in ("", "-"), (
                f"track {row['track']}: an arm beats the oracle out-of-range "
                "and no explanation is recorded"
            )


def test_injection_threshold_table(readme):
    """README's SNR-cut table must match the calibration that was run."""
    import pandas as pd

    path = get_settings().results_dir / "relativity" / "threshold_calibration.csv"
    if not path.exists():
        pytest.skip("threshold calibration not run")
    df = pd.read_csv(path)
    row = df.loc[df.snr_threshold == 2.0].iloc[0]
    assert (
        f"{abs(row.bias_pct):.0f} %" in readme or f"{abs(row.bias_pct):.0f}%" in readme
    ), f"the README's quoted bias at threshold 2.0 drifted to {row.bias_pct:.1f}%"
    best = df.loc[df.snr_threshold == 3.0].iloc[0]
    assert abs(best.bias_pct) < 5.0, (
        f"threshold 3.0 is no longer the unbiased choice: {best.bias_pct:+.1f}%"
    )


def test_injection_and_real_event_agree_on_the_pn_bias(readme):
    """README claims the injection REPRODUCES the real event's PN bias."""
    topic = _meta_topic("relativity")
    inj = topic["discovery"]["injection"]
    gw = _meta("relativity/gw150914")
    pub = gw["published_Mc_detector"]
    real = {r["pn_order"]: r for r in gw["pn_ablation"] if r.get("converged")}
    for order, key in ((0, "pn0"), (3, "pn3")):
        if key not in inj or order not in real:
            continue
        real_pct = real[order]["bias_Msun"] / pub * 100
        assert abs(inj[key]["bias_pct"] - real_pct) < 12.0, (
            f"pn{order}: injection {inj[key]['bias_pct']:+.1f}% vs real "
            f"{real_pct:+.1f}% -- no longer the same story"
        )


def _meta_topic(topic):
    import json

    p = get_settings().results_dir / topic / "problem_meta.json"
    if not p.exists():
        pytest.skip(f"{topic} topic not run")
    return json.loads(p.read_text())
