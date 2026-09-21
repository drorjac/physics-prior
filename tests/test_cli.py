"""The CLI is the product's front door: it must not crash on its own --help."""

from __future__ import annotations

import pytest

from physprior.cli import build_parser, main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "physprior" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "physprior" in capsys.readouterr().out


def test_info_reports_resolved_paths(capsys, tmp_settings):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "results" in out and "figures" in out


def test_data_list_names_every_registered_dataset(capsys):
    from physprior.data.registry import DATASETS

    assert main(["data", "list"]) == 0
    out = capsys.readouterr().out
    for name in DATASETS:
        assert name in out


def test_unknown_problem_is_rejected_not_raised():
    assert main(["run", "phlogiston"]) == 2


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
