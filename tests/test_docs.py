"""The documentation must not rot.

`docs/` is organised one folder per physics topic, and the topic pages carry
the figures. Both of those are easy to break silently by moving a file, so
they are checked rather than trusted.
"""

from __future__ import annotations

import re

import pytest

from physprior.config import get_settings
from physprior.problems import PROBLEMS

ROOT = get_settings().root
SKIP = ("http", "https", "mailto:", "#")


def _markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if any(
            part in {".venv", "node_modules", ".pytest_cache"} for part in path.parts
        ):
            continue
        yield path


def _links(text: str):
    """Markdown links and raw <img src=...>, which the topic pages use for
    side-by-side figure tables."""
    return re.findall(r"\]\(([^)#][^)]*)\)", text) + re.findall(r'src="([^"]+)"', text)


def test_every_relative_link_resolves():
    broken = []
    for path in _markdown_files():
        for target in _links(path.read_text()):
            if target.startswith(SKIP) or "<" in target:
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize("topic", PROBLEMS)
def test_every_physics_topic_has_a_docs_folder(topic):
    """One folder per topic, and it is not a stub."""
    page = ROOT / "docs" / topic / "README.md"
    assert page.exists(), f"docs/{topic}/README.md is missing"
    assert len(page.read_text().split()) > 150, f"docs/{topic} is a stub"


@pytest.mark.parametrize("topic", PROBLEMS)
def test_every_topic_page_shows_figures(topic):
    """A topic page that shows no pictures is a table of contents."""
    text = (ROOT / "docs" / topic / "README.md").read_text()
    images = [t for t in _links(text) if t.endswith((".png", ".gif"))]
    assert len(images) >= 3, f"docs/{topic} embeds only {len(images)} figures"


def test_the_docs_index_links_every_topic():
    index = (ROOT / "docs" / "README.md").read_text()
    for topic in PROBLEMS:
        assert f"({topic}/)" in index, f"docs/README.md does not link {topic}"
