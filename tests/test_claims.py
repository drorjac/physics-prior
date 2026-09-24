"""The narrative must not drift from the results.

The results TABLES in the README and docs/RESULTS.md are generated from
`results/` by `physprior report`, so they cannot drift. The handful of numbers
quoted in the prose -- "+9 Msun", "10.8 ppm", "1.13 +- 0.002" -- are written
by hand, and this file is what stops them going stale: each one is recomputed
from `results/` and checked to still appear in the README.

If one of these fails after a re-run, the number in the README is wrong and
the README is what must change.
"""

from __future__ import annotations

import json
import re

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
    """README: the 4th-order/3-hour run gives alpha = 1.1343 +- 0.0024."""
    rows = {
        (r["step"], r["fd_order"]): r
        for r in _meta("relativity/mercury")["gr_convergence"]
        if "alpha_GR" in r
    }
    bad = rows[("180m", 4)]
    assert f"{bad['alpha_GR']:.4f}" in readme, (
        f"the coarse alpha drifted to {bad['alpha_GR']:.6f}"
    )
    assert f"{bad['alpha_sigma']:.4f}" in readme, (
        f"the coarse alpha's sigma drifted to {bad['alpha_sigma']:.6f}"
    )
    assert f"{100 * (bad['alpha_GR'] - 1.0):.1f}%" in readme, (
        f"the quoted deviation drifted to {100 * (bad['alpha_GR'] - 1.0):.2f}%"
    )
    n_sigma = abs(bad["alpha_GR"] - 1.0) / bad["alpha_sigma"]
    assert f"{n_sigma:.0f} formal sigma" in readme or f"{n_sigma:.0f}σ" in readme, (
        f"the false-violation significance drifted to {n_sigma:.1f} sigma"
    )


def test_quoted_gr_numbers_reproduce_their_own_sigma_count(readme):
    """The numbers AS PRINTED must reproduce the significance AS PRINTED.

    Quoting `alpha = 1.13 +- 0.002` next to `56 formal sigma` is internally
    inconsistent -- a reader who divides gets 65 -- even though both came from
    a correct result, because alpha was rounded to 2 dp and sigma to one
    significant figure. Enough digits must survive the rounding that the
    arithmetic a reader can do is the arithmetic the pipeline did.
    """
    plain = readme.replace("**", "")
    m = re.search(r"\u03b1 = ([\d.]+) \u00b1 ([\d.]+)", plain)
    assert m, "the README no longer quotes alpha as 'alpha = X \u00b1 Y'"
    alpha_q, sigma_q = float(m.group(1)), float(m.group(2))
    quoted = re.search(r"(\d+) formal sigma", plain)
    assert quoted, "the README no longer quotes a formal sigma count"
    implied = abs(alpha_q - 1.0) / sigma_q
    assert abs(implied - int(quoted.group(1))) < 1.0, (
        f"the printed alpha = {alpha_q} \u00b1 {sigma_q} implies "
        f"{implied:.0f} sigma but the README claims {quoted.group(1)}: "
        "the quoted numbers are rounded too hard to reproduce their own claim"
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


# ---------------------------------------------------------------------------
# The neglected-terms study: the prose in docs/neglected/README.md


@pytest.fixture(scope="module")
def neglected_doc() -> str:
    """The page with its whitespace collapsed.

    Prose wraps, so a quoted phrase like "76 % to 68 %" can straddle a
    newline and a naive substring check fails on formatting rather than on
    content. Collapsing whitespace makes these tests about the numbers.
    """
    p = get_settings().root / "docs" / "neglected" / "README.md"
    if not p.exists():
        pytest.skip("docs/neglected/README.md missing")
    return " ".join(p.read_text().split())


def _neglected(name):
    import pandas as pd

    p = get_settings().results_dir / f"neglected_{name}.csv"
    if not p.exists():
        pytest.skip(f"neglected_{name}.csv missing -- run `physprior neglected`")
    return pd.read_csv(p)


def test_algebraic_crossover_numbers(neglected_doc):
    """The doc quotes the eps = 0 cost of the prior and the eps = 0.8 win."""
    d = _neglected("eps")
    g = d.groupby(["eps", "arm"]).nrmse_in.median()
    for eps in (0.0, 0.8):
        for arm in ("physics", "pinn"):
            assert f"{g[(eps, arm)]:.4f}" in neglected_doc, (
                f"eps={eps} {arm} drifted to {g[(eps, arm)]:.6f}"
            )
    ratio = g[(0.8, "physics")] / g[(0.8, "pinn")]
    assert f"{ratio:.1f}×" in neglected_doc, f"the win at eps=0.8 is now {ratio:.2f}x"


def test_noise_crossover_table(neglected_doc):
    """The doc's claim that `physics` retakes the lead by 10% noise."""
    d = _neglected("noise")
    g = d.groupby(["noise", "arm"]).nrmse_in.median()
    assert g[(0.10, "physics")] < g[(0.10, "pinn")], (
        "the noise crossover has moved: `pinn` still wins at 10% noise"
    )
    assert g[(0.0, "physics")] > g[(0.0, "pinn")], "no crossover left to describe"
    for noise in (0.0, 0.05, 0.10):
        for arm in ("physics", "pinn"):
            assert f"{g[(noise, arm)]:.4f}" in neglected_doc, (
                f"noise={noise} {arm} drifted to {g[(noise, arm)]:.6f}"
            )


def test_ode_damping_is_the_distinguishable_case(neglected_doc):
    d = _neglected("ode")
    d = d[d.amplitude_deg == 60]
    g = d.groupby(["shape", "arm"]).nrmse.median()
    ratio = g[("damping", "physics")] / g[("damping", "pinn")]
    assert ratio > 5, f"damping is no longer a decisive win: {ratio:.1f}x"
    assert f"{ratio:.0f}×" in neglected_doc, (
        f"the quoted ODE win drifted to {ratio:.1f}x"
    )
    # and the degenerate one: `physics` fits BETTER while omega is worse
    assert g[("anharmonic", "physics")] < g[("anharmonic", "pinn")], (
        "the anharmonic case no longer shows a better fit with a worse model"
    )


def test_pde_absorption_table(neglected_doc):
    """The heart of rung 3: degenerate absorbs into alpha, distinguishable does not."""
    d = _neglected("pde")
    g = d[d.arm == "physics"].groupby(["shape", "eps"]).alpha_error_pct.median()
    deg = g[("diffusive", 0.6)]
    dis = g[("advective", 0.6)]
    assert deg > 10 * dis, (
        f"the contrast collapsed: diffusive {deg:.1f}% vs advective {dis:.1f}%"
    )
    assert f"{deg:.1f} %" in neglected_doc, (
        f"the degenerate error drifted to {deg:.2f}%"
    )
    assert f"{dis:.1f} %" in neglected_doc, (
        f"the distinguishable error drifted to {dis:.2f}%"
    )


def test_doc_does_not_report_the_unconverged_pinn_as_a_result(neglected_doc):
    """Invariant 4: a non-converged arm is marked, not quoted."""
    assert "converged=False" in neglected_doc
    assert "does not depend on it" in neglected_doc


def _derivative_table():
    import pandas as pd

    p = get_settings().results_dir / "neglected" / "tune_derivative_accuracy.csv"
    if not p.exists():
        pytest.skip("run `physprior neglected derivative`")
    return pd.read_csv(p).set_index("w_smooth")


def test_derivative_accuracy_numbers(neglected_doc):
    """The diagnostic table in docs/neglected, re-derived."""
    d = _derivative_table()
    exact_uxx = float(d.mean_abs_uxx_exact.iloc[0])
    assert f"{exact_uxx:.2f}" in neglected_doc, (
        f"the exact |u_xx| drifted to {exact_uxx:.4f}"
    )
    for w in (0.0, 0.003, 0.01, 0.03):
        row = d.loc[w]
        assert f"{row.mean_abs_uxx:.2f}" in neglected_doc, (
            f"|u_xx| at w_smooth={w} drifted to {row.mean_abs_uxx:.4f}"
        )
        assert f"{row.implied_alpha:.4f}" in neglected_doc, (
            f"implied alpha at w_smooth={w} drifted to {row.implied_alpha:.6f}"
        )


def test_the_first_derivative_is_the_one_that_is_right(neglected_doc):
    """The claim the whole diagnosis rests on.

    alpha = |u_t| / |u_xx|, so alpha * |u_xx| recovers the u_t scale. If the
    network's product matches the exact one, the first derivative is right
    and the entire error is in the second -- which is what makes this a
    statement about DERIVATIVE accuracy rather than about the fit.
    """
    d = _derivative_table()
    row = d.loc[0.0]
    net = row.implied_alpha * row.mean_abs_uxx
    exact = row.alpha_true * row.mean_abs_uxx_exact
    assert abs(net / exact - 1) < 0.05, (
        f"u_t is no longer right: product {net:.4f} vs exact {exact:.4f}"
    )
    for value in (net, exact):
        assert f"{value:.4f}" in neglected_doc, (
            f"the quoted product drifted: {value:.4f}"
        )
    # and the curvature really is the thing that is wrong
    assert row.mean_abs_uxx > 1.3 * row.mean_abs_uxx_exact, (
        "the excess curvature has gone away; the doc's diagnosis no longer holds"
    )


def test_the_curvature_penalty_helps_the_least_squares_reading(neglected_doc):
    d = _derivative_table()
    err = (d.implied_alpha / d.alpha_true - 1).abs()
    assert err.loc[0.003] < err.loc[0.0] / 3, (
        f"the penalty no longer helps: {err.loc[0.0]:.3f} -> {err.loc[0.003]:.3f}"
    )
    assert f"{err.loc[0.003] * 100:.1f} %" in neglected_doc, (
        f"the best-case error drifted to {err.loc[0.003] * 100:.2f}%"
    )


def test_doc_admits_the_remedy_does_not_transfer(neglected_doc):
    """Invariant 7. The penalty works on the least-squares reading and
    largely fails inside the trained arm, and the page must say so."""
    assert "does not transfer" in neglected_doc.lower()
    import pandas as pd

    p = get_settings().results_dir / "neglected" / "tune_pde_smooth.csv"
    if not p.exists():
        pytest.skip("run `physprior neglected tune`")
    t = pd.read_csv(p)
    lo, hi = t.err_pct.min(), t.err_pct.max()
    assert lo > 50, f"the trained arm now reaches {lo:.1f}% -- the page is stale"
    assert f"{hi:.0f} % to {lo:.0f} %" in neglected_doc, (
        f"the quoted range drifted to {hi:.0f}-{lo:.0f}%"
    )
