"""The problem notebooks must follow the same four movements, in order.

    1  the problem   what is being asked, as maths and as a picture
    2  the data      created by simulation, or pulled with its provenance
    3  the method    which of the two jobs, and by which arms
    4  the results   plots, then a generated conclusion

A reader landing on any of them should meet the question before the
machinery. That is easy to lose one edit at a time, so it is checked.
"""

from __future__ import annotations

import pytest

from physprior.problems import PROBLEMS
from physprior.reporting.notebooks import BUILDERS, SECTIONS, method_block, section

PROBLEM_NOTEBOOKS = [n for n in BUILDERS if n != "00_overview"]


def _markdown(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_the_four_movements_appear_in_order(name):
    text = _markdown(BUILDERS[name]())
    positions = []
    for i, title in enumerate(SECTIONS, start=1):
        marker = f"## {title}"
        assert marker in text, f"{name} is missing section {i}: {title}"
        positions.append(text.index(marker))
    assert positions == sorted(positions), f"{name} has its sections out of order"


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_the_problem_comes_before_any_data_or_method(name):
    """The question first. A notebook that opens with a data load has buried
    the only thing a new reader needs."""
    text = _markdown(BUILDERS[name]())
    assert text.index(f"## {SECTIONS[0]}") < text.index(f"## {SECTIONS[1]}")


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_the_problem_section_states_the_law_as_maths(name):
    """Prose describing an equation is not the equation."""
    text = _markdown(BUILDERS[name]())
    problem = text[text.index(f"## {SECTIONS[0]}") : text.index(f"## {SECTIONS[1]}")]
    assert "$$" in problem, f"{name} states no governing equation in section 1"


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_the_method_section_says_which_job_is_being_done(name):
    """Discovery and recovery are different claims, and conflating them is
    the most common way to misread a physics-ML result."""
    text = _markdown(BUILDERS[name]())
    method = text[text.index(f"## {SECTIONS[2]}") : text.index(f"## {SECTIONS[3]}")]
    assert "Discovery" in method or "Recovery" in method
    assert "Arms run here:" in method


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_the_data_section_distinguishes_created_from_pulled(name):
    text = _markdown(BUILDERS[name]())
    data = text[text.index(f"## {SECTIONS[1]}") : text.index(f"## {SECTIONS[2]}")]
    assert "Created" in data and "Pulled" in data, (
        f"{name} does not say which numbers are simulated and which are measured"
    )


@pytest.mark.parametrize("name", PROBLEM_NOTEBOOKS)
def test_every_problem_notebook_ends_in_a_generated_conclusion(name):
    text = _markdown(BUILDERS[name]())
    assert "## Conclusion" in text
    assert text.index("## Conclusion") > text.index(f"## {SECTIONS[3]}")


def test_there_is_a_notebook_for_every_problem():
    for topic in PROBLEMS:
        assert any(topic in n for n in PROBLEM_NOTEBOOKS), f"no notebook for {topic}"


def test_method_block_names_both_jobs_when_both_run():
    block = method_block(discovers=True, recovers=True, arms="all five")
    assert "Discovery" in block and "Recovery" in block
    only = method_block(discovers=False, recovers=True, arms="four")
    assert "Discovery" not in only


def test_section_headings_are_canonical():
    """So the four cannot be renamed into something that only looks alike."""
    assert section(1).startswith("## 1 ")
    assert section(2, "created").endswith("— created")
