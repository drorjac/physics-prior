"""The reproduction contract.

Two halves. `physprior report --check`: the generated tables in the README
and docs/RESULTS.md are exactly what `results/` produces now. And
`physprior verify`: a regenerated results directory is compared with the
committed one, number by number, ignoring only what is meant to move.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from physprior import cli
from physprior.config import get_settings
from physprior.environment import snapshot
from physprior.reporting import report, verify


def test_generated_tables_match_the_committed_results():
    """No generated table was edited by hand, and results/ was not changed
    without re-running `physprior report`."""
    if not (get_settings().results_dir / "headline.json").is_file():
        pytest.skip("results/headline.json missing -- run `make run report`")
    stale = report.check()
    assert not stale, f"out of date, run `physprior report`: {stale}"


# --- verify -----------------------------------------------------------------


@pytest.fixture
def pair(tmp_path):
    ref, cand = tmp_path / "ref", tmp_path / "cand"
    (ref / "t").mkdir(parents=True)
    df = pd.DataFrame(
        {"arm": ["physics", "nn"], "nrmse": [1.5e-8, 0.33], "seconds": [0.1, 9.0]}
    )
    df.to_csv(ref / "t" / "sweep.csv", index=False)
    (ref / "t" / "meta.json").write_text(
        json.dumps({"GM": 1.3272e20, "fit": {"seconds": 1.0, "n": 4}})
    )
    shutil.copytree(ref, cand)
    return ref, cand


def _status(results, name):
    return next(r for r in results if r.path == name).status


def test_an_unchanged_copy_is_identical(pair):
    ref, cand = pair
    assert {r.status for r in verify.compare_dirs(ref, cand)} == {"identical"}


def test_wall_clock_time_is_allowed_to_move(pair):
    ref, cand = pair
    df = pd.read_csv(cand / "t" / "sweep.csv")
    df["seconds"] = [5.0, 50.0]
    df.to_csv(cand / "t" / "sweep.csv", index=False)
    (cand / "t" / "meta.json").write_text(
        json.dumps({"GM": 1.3272e20, "fit": {"seconds": 7.0, "n": 4}})
    )
    results = verify.compare_dirs(ref, cand)
    assert _status(results, "t/sweep.csv") == "close"
    assert _status(results, "t/meta.json") == "close"
    assert not any(r.failed for r in results)


def test_a_moved_number_is_caught(pair):
    ref, cand = pair
    df = pd.read_csv(cand / "t" / "sweep.csv")
    df.loc[0, "nrmse"] *= 1 + 1e-6
    df.to_csv(cand / "t" / "sweep.csv", index=False)
    r = next(r for r in verify.compare_dirs(ref, cand) if r.path == "t/sweep.csv")
    assert r.failed
    assert "nrmse[0]" in r.detail


def test_a_moved_constant_inside_json_is_caught(pair):
    ref, cand = pair
    (cand / "t" / "meta.json").write_text(
        json.dumps({"GM": 1.3273e20, "fit": {"seconds": 1.0, "n": 4}})
    )
    assert _status(verify.compare_dirs(ref, cand), "t/meta.json") == "differs"


def test_a_changed_label_is_caught(pair):
    ref, cand = pair
    df = pd.read_csv(cand / "t" / "sweep.csv")
    df.loc[1, "arm"] = "pinn"
    df.to_csv(cand / "t" / "sweep.csv", index=False)
    assert _status(verify.compare_dirs(ref, cand), "t/sweep.csv") == "differs"


def test_missing_and_new_files_are_listed_not_failed(pair):
    ref, cand = pair
    (cand / "t" / "meta.json").unlink()
    (cand / "extra.csv").write_text("a\n1\n")
    (cand / "environment.json").write_text("{}")
    results = verify.compare_dirs(ref, cand)
    assert _status(results, "t/meta.json") == "not regenerated"
    assert _status(results, "extra.csv") == "new"
    assert all(r.path != "environment.json" for r in results)
    assert not any(r.failed for r in results)


def test_the_cli_exits_nonzero_on_a_difference(pair, capsys):
    ref, cand = pair
    assert cli.main(["-q", "verify", str(cand), "--reference", str(ref)]) == 0
    (cand / "t" / "meta.json").write_text(json.dumps({"GM": 1.0, "fit": {"n": 4}}))
    assert cli.main(["-q", "verify", str(cand), "--reference", str(ref)]) == 1
    assert "differs" in capsys.readouterr().out


# --- environment ------------------------------------------------------------


def test_the_environment_record_names_what_can_move_a_number():
    env = snapshot()
    for key in ("physprior", "git_commit", "python", "packages", "torch_device"):
        assert key in env
    assert env["packages"]["numpy"]
    json.dumps(env)  # must be writable as-is
