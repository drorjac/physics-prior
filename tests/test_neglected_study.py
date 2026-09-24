"""The study's outputs must be reproducible by a command.

This file exists because they were not. Five CSVs under `results/` and
fourteen figures under `figures/neglected/` were committed, and nothing in
the repository regenerated them: `scripts/regenerate.sh` ran every problem,
wrote every other figure, rebuilt the docs and the notebooks, and skipped
this study. A committed artefact that no command reproduces cannot be
checked, and silently stops matching the code that supposedly made it.
"""

from __future__ import annotations

import pytest

from physprior.reporting import neglected_study as NS

STAGES = ("algebraic", "ode", "pde", "derivative", "tune")


def test_every_stage_is_reachable():
    import inspect

    src = inspect.getsource(NS.main)
    for name in STAGES:
        assert f'"{name}"' in src, f"stage {name} is not in main()'s table"
        assert callable(getattr(NS, name.replace("derivative", "derivative_accuracy")))


def test_unknown_stage_fails_loudly():
    with pytest.raises(SystemExit, match="no stage matches"):
        NS.main(only="does-not-exist")


def test_regenerate_script_runs_the_study():
    """The gap that started this file: the pipeline skipped the study."""
    from physprior.config import get_settings

    script = (get_settings().root / "scripts" / "regenerate.sh").read_text()
    assert "physprior neglected" in script, (
        "scripts/regenerate.sh does not regenerate the neglected-terms study, "
        "so its committed CSVs and figures are orphaned again"
    )


def test_cli_exposes_the_study():
    from physprior.cli import build_parser

    args = build_parser().parse_args(["neglected", "tune", "--quick"])
    assert args.command == "neglected"
    assert args.stage == "tune"
    assert args.quick is True


@pytest.mark.parametrize("name", ["eps", "noise", "budget", "ode", "pde"])
def test_committed_tables_have_the_columns_the_figures_read(name):
    import pandas as pd

    from physprior.config import get_settings

    path = get_settings().results_dir / f"neglected_{name}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} absent -- run `physprior neglected`")
    df = pd.read_csv(path)
    assert "seed" in df and "arm" in df
    metric = "nrmse" if name in ("ode", "pde") else "nrmse_in"
    assert metric in df.columns, f"{path.name} lost its {metric} column"
    assert len(df) > 0


def test_every_committed_figure_is_regenerated_by_some_stage():
    """No orphans. A figure in the repo that no stage writes is drift waiting
    to happen: it cannot be checked, and nothing notices when the code that
    made it changes.

    The first version of the driver covered nine of the fourteen and missed
    the six single-run detail figures, which is exactly the gap this asserts.
    """
    import inspect

    from physprior.config import get_settings

    figures = get_settings().figures_dir / "neglected"
    if not figures.exists():
        pytest.skip("figures/neglected absent")
    committed = {p.stem for p in figures.glob("*.png")}
    assert committed, "no committed figures to check"

    written = inspect.getsource(NS)
    missing = sorted(n for n in committed if f'"{n}"' not in written)
    assert not missing, (
        "these committed figures are not written by any stage of "
        f"`physprior neglected`: {missing}"
    )


def test_cli_help_lists_exactly_the_stages_that_exist():
    """A help string that names a stage which does not exist, or omits one
    that does, is a small lie the user pays for."""
    import inspect
    import re

    from physprior.cli import build_parser

    table = re.findall(r'^\s+"(\w+)":', inspect.getsource(NS.main), re.M)
    assert table, "could not read main()'s stage table"

    action = next(
        a
        for a in build_parser()
        ._subparsers._group_actions[0]
        .choices["neglected"]
        ._actions
        if a.dest == "stage"
    )
    advertised = {w.strip() for w in re.split(r"[|]", action.help.split("(")[0])}
    assert advertised == set(table), (
        f"help advertises {sorted(advertised)} but main() has {sorted(table)}"
    )
