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
from physprior.io import load_json, load_table


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


def _sci(x: float, digits: int) -> str:
    """1.2e-04 -> '1.2×10⁻⁴', the way the prose prints it."""
    mant, exp = f"{x:.{digits}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mant}×10{str(int(exp)).translate(sup)}"


def test_mercury_residual_is_two_omissions_that_nearly_cancel(readme):
    """README + docs/relativity: what is left of alpha - 1 once it has converged.

    Four physical statements, each re-derived:
    - the residual is much larger than what the derivative can still move;
    - putting back DE441's own solar J2 and frame dragging moves alpha AWAY
      from 1, and the n-body (EIH) term alone moves it the other way;
    - both together -- DE441's own model -- give alpha = 1 within 2 sigma
      at both steps; if that ever fails, the prose's conclusion is wrong;
    - the quoted numbers are the recorded ones.
    """
    meta = _meta("relativity/mercury")
    if "neglected_terms" not in meta:
        pytest.skip("relativity/mercury predates neglected_terms -- rerun it")
    tr = meta["truncation"]
    resid = meta["gr"]["alpha_GR"] - 1.0
    assert abs(tr["fine_truncation_estimate"]) < 0.01 * abs(resid), (
        "the derivative's remaining error is no longer negligible against the "
        "residual: the 'not numerical' conclusion does not hold"
    )
    rows = {
        (r["step"], r["model"]): r["alpha_GR"] - 1.0 for r in meta["neglected_terms"]
    }
    sigmas = {
        (r["step"], r["model"]): r["alpha_minus_one_sigmas"]
        for r in meta["neglected_terms"]
    }
    for step in ("180m", "90m"):
        assert rows[(step, "minus_J2_and_LT")] > rows[(step, "as_shipped")] > 0, (
            f"at {step}, removing the Sun's J2 and frame dragging no longer "
            "moves alpha away from 1 -- rewrite the Mercury paragraph"
        )
        assert rows[(step, "EIH")] < 0, f"at {step}, EIH alone no longer overshoots"
        assert abs(sigmas[(step, "EIH_minus_J2_and_LT")]) < 2, (
            f"at {step}, DE441's own model no longer gives alpha = 1 within 2 sigma"
        )
    relativity = (get_settings().root / "docs" / "relativity" / "README.md").read_text()
    for text in (" ".join(readme.split()), " ".join(relativity.split())):
        assert f"+{_sci(resid, 1)}" in text
        assert f"+{_sci(rows[('180m', 'minus_J2_and_LT')], 1)}" in text
        assert _sci(rows[("180m", "EIH")], 1).replace("-", "−") in text
        full = next(
            r
            for r in meta["neglected_terms"]
            if r["step"] == "180m" and r["model"] == "EIH_minus_J2_and_LT"
        )
        assert (
            f"+{_sci(full['alpha_GR'] - 1.0, 1)} ± {_sci(full['alpha_sigma'], 1)}"
            in text
        )
    assert f"by only {_sci(tr['step_change'], 0)}" in " ".join(readme.split())
    assert f"about {_sci(tr['fine_truncation_estimate'], 1)}" in " ".join(
        relativity.split()
    )


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


def test_extrapolation_gap_holds_on_every_seed(headline):
    """The same claim, paired by seed: a median can hide a seed that
    disagrees, so the gap must exceed 100x on each reporting seed."""
    tracks = {r["track"] for r in headline["extrapolation_summary"]}
    for tr in ("quantum/cmb", "quantum/hydrogen", "gravity/kepler"):
        if tr not in tracks:
            continue
        wide = load_table(tr, "extrapolation").pivot_table(
            index="seed", columns="arm", values="nrmse_out"
        )
        ratio = wide["nn"] / wide["physics"]
        assert (ratio > 100).all(), (
            f"track {tr}: nn/physics out-of-range ratio per seed is "
            f"{ratio.round(1).to_dict()}"
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


def test_the_correction_network_hypothesis_stays_refuted(neglected_doc):
    """A negative result is a result, and this one must not quietly rot.

    The hypothesis was that the free correction C(u, u_x) destroys the
    identifiability of alpha, since (alpha, C) admits any alpha. Tested at
    eps = 0 where the true C is exactly zero: penalising C over six decades
    and then deleting it outright does not help, and removing it is slightly
    WORSE. If a future change makes removing C help, this test fails and the
    prose has to be rewritten -- which is the point.
    """
    import pandas as pd

    p = get_settings().results_dir / "neglected" / "tune_pde_correction.csv"
    if not p.exists():
        pytest.skip("run `physprior neglected tune`")
    d = pd.read_csv(p).set_index("setting")
    free = d[d.index.str.contains("free")].iloc[0]
    removed = d[d.index.str.contains("removed", case=False)].iloc[0]
    assert removed.err_pct >= free.err_pct - 5.0, (
        f"removing C now helps ({free.err_pct:.1f}% -> {removed.err_pct:.1f}%): "
        "the refutation in docs/METHOD.md no longer holds"
    )
    assert "Refuted" in neglected_doc or "refuted" in neglected_doc
    for row in (free, removed):
        assert f"{row.alpha_mean:.5f}" in neglected_doc, (
            f"a quoted alpha drifted to {row.alpha_mean:.6f}"
        )


def test_the_optimiser_is_self_consistent(neglected_doc):
    """The measurement the whole PDE diagnosis rests on.

    With C removed, the trained alpha must equal the least-squares alpha of
    the field the training produced. If it does, the optimiser is doing what
    it was asked and the error is entirely in the FIELD -- which is what
    makes "the physics term degrades the field" a conclusion rather than a
    guess. If a future change breaks that equality, the diagnosis in
    docs/METHOD.md no longer follows and this test says so.
    """
    import pandas as pd

    p = get_settings().results_dir / "neglected" / "tune_pde_consistency.csv"
    if not p.exists():
        pytest.skip("run `physprior neglected consistency`")
    d = pd.read_csv(p)

    removed = d[d.setting.str.contains("removed")]
    assert len(removed) >= 3
    assert (removed.ratio > 0.97).all() and (removed.ratio < 1.03).all(), (
        "the optimiser is no longer self-consistent with C removed: ratios "
        f"{removed.ratio.round(3).tolist()}"
    )

    # invariant 5: the finite-difference reading must be converged
    assert (d.fd_drift < 1e-4).all(), (
        f"the derivative estimate is still moving with h: max drift {d.fd_drift.max():.1e}"
    )

    # and the contrast that shows C breaks the consistency
    free = d[d.setting.str.contains("free")]
    assert free.ratio.max() < 0.8, (
        f"C no longer absorbs the discrepancy: ratios {free.ratio.round(2).tolist()}"
    )

    for value in removed.ratio:
        assert f"{value:.2f}×" in neglected_doc, (
            f"a quoted consistency ratio drifted to {value:.3f}"
        )


def test_the_residual_makes_the_field_worse(neglected_doc):
    """The conclusion itself: turning the physics term on costs field
    accuracy. The data-only field implies an alpha closer to the truth than
    the residual-trained field does."""
    import pandas as pd

    pc = get_settings().results_dir / "neglected" / "tune_pde_consistency.csv"
    pd_ = get_settings().results_dir / "neglected" / "tune_derivative_accuracy.csv"
    if not (pc.exists() and pd_.exists()):
        pytest.skip("run `physprior neglected consistency derivative`")
    trained = pd.read_csv(pc)
    trained = trained[trained.setting.str.contains("removed")].field_implies.mean()
    data_only = pd.read_csv(pd_).set_index("w_smooth").loc[0.0].implied_alpha
    true = 0.05
    assert abs(data_only - true) < abs(trained - true), (
        "the residual no longer degrades the field: data-only implies "
        f"{data_only:.4f}, residual-trained {trained:.4f}, true {true}"
    )
    assert f"{data_only:.3f}" in neglected_doc


# --- quantum/helium: docs/quantum/README.md ----------------------------------


@pytest.fixture(scope="module")
def quantum_doc() -> str:
    return (get_settings().root / "docs" / "quantum" / "README.md").read_text()


@pytest.fixture(scope="module")
def helium() -> dict:
    root = get_settings().results_dir / "quantum" / "helium"
    if not (root / "meta.json").is_file():
        pytest.skip("results/quantum/helium missing -- run the quantum problem")
    import pandas as pd

    ex = pd.read_csv(root / "extrapolation.csv")
    return {
        "meta": json.loads((root / "meta.json").read_text()),
        "out": ex.pivot_table(index="seed", columns="arm", values="nrmse_out"),
        "pinn": pd.read_csv(root / "pinn_correction.csv"),
    }


def test_helium_rydberg_absorbs_the_defect(helium, quantum_doc):
    m = helium["meta"]
    dev = (m["headline"]["physics"]["params"]["R"] / m["rydberg_he_icm"] - 1) * 100
    assert f"{dev:.1f}% above" in quantum_doc
    ratio = float((helium["out"]["physics"] / helium["out"]["oracle"]).median())
    assert f"{ratio:.0f} times worse" in quantum_doc


def test_helium_correction_hurts_out_of_range(helium, quantum_doc):
    p = helium["pinn"]
    tuned = p[p.variant == "pinn"]
    assert (tuned.correction_rms_frac > 0.1).all()
    worse = helium["out"]["pinn"] / helium["out"]["physics"]
    assert (worse > 1).all()
    assert f"{worse.min():.1f} to {worse.max():.1f} times worse" in quantum_doc
    balanced = p[p.variant == "pinn_balanced"]
    assert (balanced.correction_rms_frac < 2e-6).all()
    assert "below 2×10⁻⁶" in quantum_doc
    gap = (
        (balanced.set_index("seed").nrmse_out / helium["out"]["physics"] - 1)
        .abs()
        .max()
    )
    assert gap < 0.003


def test_helium_sr_beats_the_hydrogenic_law_on_every_seed(helium, quantum_doc):
    gain = helium["out"]["physics"] / helium["out"]["sr"]
    assert (gain > 1).all()
    assert f"factor of {gain.min():.1f} to {gain.max():.1f}" in quantum_doc


def test_helium_rydberg_ritz_error(helium, quantum_doc):
    err = helium["meta"]["rydberg_ritz"]["nrmse_out"]
    mant, exp = f"{err:.1e}".split("e")
    sup = str(int(exp[1:])).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    assert f"{mant}×10⁻{sup}" in quantum_doc


# --- gravity/pulsar_spindown: docs/gravity/README.md --------------------------


@pytest.fixture(scope="module")
def pulsar() -> dict:
    root = get_settings().results_dir / "gravity" / "pulsar_spindown"
    if not (root / "meta.json").is_file():
        pytest.skip(
            "results/gravity/pulsar_spindown missing -- run the gravity problem"
        )
    return json.loads((root / "meta.json").read_text())


@pytest.fixture(scope="module")
def gravity_doc() -> str:
    return (get_settings().root / "docs" / "gravity" / "README.md").read_text()


def test_pulsar_dipole_deficit(pulsar, gravity_doc):
    d = pulsar["dipole_test"]
    assert f"n = {d['n_fit']:.2f} ± {d['n_sigma']:.2f}" in gravity_doc
    assert f"{d['deficit_sigma']:.1f}σ below 3" in gravity_doc
    assert f"{pulsar['n_selected']} pass a rule" in gravity_doc


def test_pulsar_predictions_hold_as_written(pulsar, gravity_doc):
    v = pulsar["verdicts"]
    assert all(v[k] for k in v if k.startswith("p"))
    out = v["median_nrmse_out"]
    assert f"{min(out.values()):.2f} to {max(out.values()):.2f} times" in gravity_doc
    sim = list(v["pinn_over_physics_sim"].values())
    assert f"{min(sim):.2f} to {max(sim):.2f} of the constant-n fit" in gravity_doc


# --- H7: the physics weight against the data budget ---------------------------


def test_h7_verdict_as_quoted():
    from physprior.benchmark import budget_weight as B

    root = get_settings().results_dir
    if not (root / "quantum" / "helium" / "budget_weight.csv").is_file():
        pytest.skip("H7 tables missing -- run physprior.benchmark.budget_weight")
    v = B.verdicts()
    assert not v["refuted"]
    text = (get_settings().root / "docs" / "HYPOTHESES.md").read_text()
    held = len(v["p1_weight_shrinks_or_holds"]) - v["p1_failures"]
    assert f"holds on {held} of {len(v['p1_weight_shrinks_or_holds'])} tracks" in text
    assert f"holds at {v['p2_cells_no_worse']} of {v['p2_cells']} cells" in text
    assert not v["p1_weight_shrinks_or_holds"]["quantum/cmb"]
