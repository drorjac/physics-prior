"""Logging.

A library must not print. It emits records on the `physprior` logger and lets
the application decide what to do with them; the CLI is the only place that
configures a handler.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT_NAME = "physprior"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """A child logger. Pass `__name__` from inside the package."""
    if not name or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def configure(level: int | str | None = None, *, stream=None) -> None:
    """Attach a handler to the package logger. Called by the CLI, not on import.

    `PHYSPRIOR_LOG_LEVEL` overrides the level when nothing is passed.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_NAME)
    if level is None:
        level = os.environ.get("PHYSPRIOR_LOG_LEVEL", "INFO")
    logger.setLevel(level)
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
