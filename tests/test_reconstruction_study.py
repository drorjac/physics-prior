"""reconstruction.study: the summaries, and a quick end-to-end run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physprior.reconstruction import study


def _sweep(err_by_method):
    rows = []
    for (case, d, m), errs in err_by_method.items():
        for n, e in errs.items():
            for s in (11, 23, 42):
                rows.append(
                    {"case": case, "d": d, "method": m, "N": n, "seed": s, "nrmse": e}
                )
    return pd.DataFrame(rows)


def test_samples_needed_interpolates_and_censors():
    df = _sweep(
        {
            ("box", 1, "physics"): {8: 0.2, 16: 0.05},
            ("box", 1, "gp"): {8: 0.5, 16: 0.3},
            ("box", 1, "nn"): {8: 0.05, 16: 0.01},
        }
    )
    ns = study.samples_needed(df, 0.1).set_index("method")
    # log-linear between (8, 0.2) and (16, 0.05): 0.1 is half-way in log
    assert ns.loc["physics", "N_star"] == pytest.approx(8 * np.sqrt(2))
    assert ns.loc["gp", "censored"]
    assert ns.loc["nn", "at_first_grid_point"]


def test_verdict_reads_the_three_criteria():
    errs = {}
    for d, (p, b) in {1: (8, 16), 2: (12, 128), 3: (16, 1024)}.items():
        errs[("box", d, "physics")] = {p: 0.05}
        for m in ("gp", "interp", "nn"):
            errs[("box", d, m)] = {b: 0.05}
    ns = study.samples_needed(_sweep(errs), 0.1)
    v = study.verdict(ns)["box"]
    assert v["criterion_1_black_box_grows"]
    assert v["criterion_2_physics_tracks_unknowns"]
    assert v["criterion_3_gap_widens"]
    assert v["supported"]


@pytest.mark.slow
def test_quick_run_end_to_end(tmp_settings):
    meta = study.run(quick=True, workers=1)
    assert meta["quick"]
    res = tmp_settings.results_dir / "reconstruction"
    for name in ("sweep.csv", "samples_needed.csv", "verdict.json", "README.md"):
        assert (res / name).exists()
    assert (tmp_settings.figures_dir / "reconstruction" / "samples_needed.png").exists()
