"""Run the fields problem.

physprior run fields

Each sub-study is a (name, module) pair whose module has `run(quick)`. A new
study is added by appending to `SUBRUNS`; modules are imported only when run,
so one study's optional dependencies do not block the others.
"""

from __future__ import annotations

import importlib
from typing import Any

from physprior.io import save_json

PROBLEM = "fields"

SUBRUNS: list[tuple[str, str]] = [
    ("weather", "physprior.problems.fields.weather"),
    ("rf", "physprior.problems.fields.rf"),
]


def run(quick: bool = False, only: str | None = None) -> dict:
    meta: dict[str, Any] = {"problem": PROBLEM}
    for name, module in SUBRUNS:
        if only and only != name:
            continue
        meta[name] = importlib.import_module(module).run(quick=quick)
    save_json(meta, PROBLEM, "problem_meta")
    return meta
