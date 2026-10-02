"""Exception hierarchy.

Every failure this package raises is one of these, so a caller can tell a
missing download apart from a unit slip apart from a fit that never
converged. The library never raises a bare `AssertionError` for a condition a
user can hit -- `assert` statements vanish under `python -O`, which is exactly
the wrong behaviour for a guard that protects a physical result.
"""

from __future__ import annotations


class PhysPriorError(Exception):
    """Base class for every error raised by this package."""


class DataError(PhysPriorError):
    """A dataset could not be obtained, parsed, or trusted."""


class DownloadError(DataError):
    """A remote source was unreachable or returned something unusable.

    `status` is the HTTP status code when the server answered with one, so a
    caller can tell "not found" apart from an outage without parsing text.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class UnitError(DataError):
    """A quantity is not in the units its loader promises.

    This project mixes cm^-1 with m^-1, MJy/sr with SI, AU/day with m/s, and
    GPS seconds with seconds-before-merger. A silent unit slip is
    indistinguishable from a discovery, so every loader asserts its units and
    raises this when one fails.
    """


class ConvergenceError(PhysPriorError):
    """A fit or an integration did not converge, or converged to a bound.

    A parameter pinned to its bound has not converged whatever the optimiser
    reports, and a numerical result that is still moving with step size is not
    a result.
    """


class ConfigurationError(PhysPriorError):
    """The package is misconfigured -- bad path, missing optional dependency."""


class MissingDependencyError(ConfigurationError):
    """An optional extra is needed for this code path."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"{package!r} is required for this operation. "
            f"Install it with:  pip install 'physprior[{extra}]'"
        )
        self.package = package
        self.extra = extra
