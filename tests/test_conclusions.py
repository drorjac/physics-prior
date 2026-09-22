"""The notebook conclusions must not overstate what the results support.

Two failure modes are guarded here, both of which this module committed once:

  * rounding a gap smaller than the seed-to-seed spread into a "win";
  * scoring a sweep on `nrmse_in`, which in `sweep_budget` and `sweep_noise`
    is the fit to the TRAINING points, and so rewards exact interpolation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from physprior.reporting import conclusions as C

SEEDS = (11, 23, 42)


def _frame(track, arms, wobbles=(0.0, 0.0, 0.0)):
    """`arms` maps an arm name to (nrmse_in, nrmse_out); `wobbles` shifts
    nrmse_out per seed, which is how a seed spread is simulated."""
    rows = []
    for seed, wobble in zip(SEEDS, wobbles):
        for arm, (n_in, n_out) in arms.items():
            rows.append(
                {
                    "track": track,
                    "arm": arm,
                    "n_train": 4,
                    "seed": seed,
                    "nrmse_in": n_in,
                    "nrmse_out": n_out + wobble,
                }
            )
    return pd.DataFrame(rows)


def _write(settings, track, name, df):
    out = settings.results_dir / track
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)


def _question(track, name="interpolation"):
    return next(v for v in C.verdicts(track) if v.question == name)


def test_verdict_reads_held_out_error_not_training_error(tmp_settings):
    """The regression test for the bug that made `physics` beat `sr` by
    4.7e6x on two points: that was one parameter fitted through them.

    `physics` here fits its training points exactly and generalises badly;
    `pinn` does the opposite. The winner names the column that was read.
    """
    _write(
        tmp_settings,
        "t/x",
        "sweep_budget",
        _frame("t/x", {"physics": (1e-14, 1.0), "pinn": (1.0, 1e-3)}),
    )
    assert _question("t/x").winner == "pinn", (
        "interpolation was scored on nrmse_in (the training fit); it must "
        "use the held-out nrmse_out"
    )


def test_gap_inside_the_seed_spread_is_a_tie(tmp_settings):
    _write(
        tmp_settings,
        "t/y",
        "sweep_budget",
        _frame(
            "t/y",
            {"physics": (0.0, 1.0), "pinn": (0.0, 1.1)},
            wobbles=(-0.4, 0.0, 0.4),
        ),
    )
    v = _question("t/y")
    assert not v.decisive, "a 0.1 gap under a 0.4 seed spread is not a win"
    assert "TIE" in v.as_row()["verdict"]


def test_a_gap_larger_than_the_spread_is_decisive(tmp_settings):
    _write(
        tmp_settings,
        "t/z",
        "sweep_budget",
        _frame("t/z", {"physics": (0.0, 0.01), "pinn": (0.0, 1.0)}),
    )
    v = _question("t/z")
    assert v.decisive and v.winner == "physics"
    assert pytest.approx(v.margin, rel=1e-6) == 100.0


def test_no_decisive_question_says_so_instead_of_picking_a_winner(tmp_settings):
    _write(
        tmp_settings,
        "t/w",
        "sweep_budget",
        _frame(
            "t/w",
            {"physics": (0.0, 1.0), "pinn": (0.0, 1.02)},
            wobbles=(-0.5, 0.0, 0.5),
        ),
    )
    text = C.conclusion_markdown("t/w")
    assert "No conclusion is available" in text
    assert "indistinguishable" in text
    assert C.what_would_change_this("t/w")


def test_missing_results_do_not_invent_a_conclusion(tmp_settings):
    assert "No conclusion" in C.conclusion_markdown("t/absent")
    assert C.verdict_frame("t/absent").empty


def test_the_oracle_is_not_a_competitor(tmp_settings):
    """Invariant 8: the oracle bounds the others, it does not race them."""
    _write(
        tmp_settings,
        "t/o",
        "sweep_budget",
        _frame(
            "t/o",
            {"oracle": (0.0, 1e-9), "physics": (0.0, 0.01), "pinn": (0.0, 1.0)},
        ),
    )
    assert _question("t/o").winner == "physics", (
        "the oracle must not be entered as a competitor"
    )


def test_real_results_produce_a_conclusion_for_every_scope():
    """On the committed results, each notebook's scope must render."""
    for scope in ("all", "gravity", "relativity", "quantum"):
        if C.verdict_frame(scope).empty:
            pytest.skip(f"{scope} not run yet")
        text = C.conclusion_markdown(scope)
        assert len(text.split()) > 30
        assert "{" not in text, "an unrendered template variable leaked"
        assert C.what_would_change_this(scope)


def test_no_verdict_rests_on_an_exact_interpolation():
    """A margin of millions means an arm reproduced its own training points,
    not that it won. If one reappears, the metric slipped back."""
    frame = C.verdict_frame("all")
    if frame.empty:
        pytest.skip("results not on disk")
    margins = []
    for text in frame["margin"]:
        try:
            margins.append(float(str(text).rstrip("x")))
        except ValueError:
            continue
    assert margins, "no numeric margins parsed"
    assert max(margins) < 1e5, (
        f"a margin of {max(margins):.3g}x is exact interpolation, not a result"
    )
