"""The figure palette is checkable, so it is checked.

The reference implementation of this method is JavaScript; this port must
reproduce its published numbers, or the gate it enforces means nothing.
"""

from __future__ import annotations

import pytest

from physprior.viz.palette import ARM_PALETTE, contrast, dE, validate

REFERENCE_EIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


def test_port_reproduces_the_reference_numbers():
    """Documented: worst adjacent CVD dE 9.1, worst normal-vision dE 19.6."""
    worst_cvd = min(
        dE(REFERENCE_EIGHT[i], REFERENCE_EIGHT[i + 1], k)
        for k in ("protan", "deutan")
        for i in range(len(REFERENCE_EIGHT) - 1)
    )
    worst_normal = min(
        dE(REFERENCE_EIGHT[i], REFERENCE_EIGHT[i + 1])
        for i in range(len(REFERENCE_EIGHT) - 1)
    )
    assert round(worst_cvd, 1) == 9.1
    assert round(worst_normal, 1) == 19.6


def test_the_arm_palette_passes_all_pairs_on_the_light_surface():
    """Bar charts and small multiples put every pair on screen together."""
    _, ok = validate(ARM_PALETTE, mode="light", pairs="all")
    assert ok


def test_a_known_bad_pair_is_rejected():
    """Orange beside red fails for a deuteranope; the gate must catch it."""
    _, ok = validate(["#2a78d6", "#eb6834", "#e34948"], mode="light", pairs="all")
    assert not ok


@pytest.mark.parametrize("colour", ARM_PALETTE)
def test_contrast_is_reported_not_assumed(colour):
    assert contrast(colour, "#fcfcfb") > 1.0
