"""Unit and sanity guards.

The project simultaneously handles cm^-1 and m^-1, MJy/sr and
W m^-2 sr^-1 Hz^-1, AU/day and m/s, GPS seconds and seconds-before-merger, and
solar masses and seconds. A silent unit slip in any of them would be
indistinguishable from a discovery, so every loader states the range it
promises and this module enforces it.

`require` is used rather than `assert` on purpose: assertions are removed by
`python -O`, and a guard that disappears under optimisation is worse than no
guard at all.
"""

from __future__ import annotations

from typing import TypeVar

from .exceptions import UnitError

T = TypeVar("T")


def require(condition: bool, message: str) -> None:
    """Raise `UnitError` unless `condition` holds."""
    if not condition:
        raise UnitError(message)


def require_range(
    value: float, low: float, high: float, what: str, unit: str = ""
) -> None:
    """Raise unless `low <= value <= high`."""
    if not low <= value <= high:
        suffix = f" {unit}" if unit else ""
        raise UnitError(
            f"{what} is {value:g}{suffix}, outside the expected "
            f"[{low:g}, {high:g}]{suffix}"
        )


def require_not_none(value: T | None, message: str) -> T:
    """Return `value`, or raise `UnitError` if it is None.

    `require(x is not None, ...)` guards at runtime but tells a type checker
    nothing, so every later use of `x` is still `T | None`. Returning the
    value narrows it for both the reader and the checker.
    """
    if value is None:
        raise UnitError(message)
    return value
