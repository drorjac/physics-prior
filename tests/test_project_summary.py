"""SUMMARY.md is generated; it must match a fresh render of results/."""

from __future__ import annotations

import pytest

from physprior.config import get_settings


def _ready() -> bool:
    root = get_settings().root
    return (root / "SUMMARY.md").exists() and (
        get_settings().results_dir / "lorenz" / "main.csv"
    ).exists()


@pytest.mark.skipif(not _ready(), reason="SUMMARY.md or results/lorenz missing")
def test_summary_page_is_up_to_date(tmp_path):
    from physprior.reporting.project_summary import PAGE, render

    fresh = render(tmp_path / PAGE)
    committed = (get_settings().root / PAGE).read_text()
    assert fresh == committed, "SUMMARY.md is stale: run `physprior summary-md`"


@pytest.mark.skipif(not _ready(), reason="SUMMARY.md or results/lorenz missing")
def test_summary_links_resolve():
    import re

    root = get_settings().root
    text = (root / "SUMMARY.md").read_text()
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith("http"):
            continue
        assert (root / target).exists(), f"SUMMARY.md links to missing {target}"
