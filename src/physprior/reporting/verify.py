"""Does a regenerated `results/` match the committed one?

`physprior verify CANDIDATE`

The pipeline is deterministic: every seed is fixed, the symbolic search runs
`deterministic=True`, and a full re-run has reproduced the committed tables
to every printed digit. So the comparison is numeric and tight. What it
ignores is what is *meant* to move between runs:

- columns and keys named `seconds` (wall-clock time);
- `environment.json`, which records the machine that produced the run.

Each file gets one verdict:

    identical        byte for byte
    close            same shape, every number within rtol/atol
    differs          a number, a string or the shape changed -> exit 1
    not regenerated  committed, but the candidate run did not write it
    new              written by the candidate, not committed

`not regenerated` is not a failure: `physprior run all` does not re-run the
tuning studies (`physprior tune`) or `physprior report`, so their outputs are
listed rather than faulted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IGNORED_FILES = frozenset({"environment.json"})
IGNORED_FIELDS = frozenset({"seconds"})

RTOL = 1e-9
# Purely relative by default. Errors here run down to 1e-9 (a Kepler fit's
# in-range nrmse), so any absolute floor large enough to matter would pass a
# number that had moved by orders of its own size.
ATOL = 0.0


@dataclass(frozen=True)
class FileResult:
    path: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "differs"


def _files(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name not in IGNORED_FILES and not p.name.startswith(".")
    }


def _compare_csv(ref: Path, cand: Path, rtol: float, atol: float) -> tuple[bool, str]:
    a, b = pd.read_csv(ref), pd.read_csv(cand)
    a = a.drop(columns=[c for c in a.columns if c in IGNORED_FIELDS])
    b = b.drop(columns=[c for c in b.columns if c in IGNORED_FIELDS])
    if list(a.columns) != list(b.columns):
        return False, f"columns {list(a.columns)} -> {list(b.columns)}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} -> {b.shape}"
    worst, where = 0.0, ""
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            ok = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                i = int(np.flatnonzero(~ok)[0])
                return False, f"{col}[{i}]: {float(xv[i])!r} -> {float(yv[i])!r}"
            finite = np.isfinite(xv) & np.isfinite(yv) & (xv != 0)
            if finite.any():
                rel = float(
                    np.max(np.abs(yv[finite] - xv[finite]) / np.abs(xv[finite]))
                )
                if rel > worst:
                    worst, where = rel, col
        else:
            xs, ys = x.fillna("").astype(str), y.fillna("").astype(str)
            if not (xs == ys).all():
                i = int(np.flatnonzero((xs != ys).to_numpy())[0])
                return False, f"{col}[{i}]: {xs.iloc[i]!r} -> {ys.iloc[i]!r}"
    if worst:
        return True, f"max rel. diff {worst:.1e} in {where}"
    return True, "numbers equal; only timing or formatting changed"


def _walk(a: Any, b: Any, rtol: float, atol: float, at: str = "") -> str | None:
    """The first difference between two JSON values, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = {k for k in a if k not in IGNORED_FIELDS}
        keys_b = {k for k in b if k not in IGNORED_FIELDS}
        if keys_a != keys_b:
            return f"{at or '/'}: keys {sorted(keys_a ^ keys_b)} differ"
        for k in sorted(keys_a):
            d = _walk(a[k], b[k], rtol, atol, f"{at}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{at}: length {len(a)} -> {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _walk(x, y, rtol, atol, f"{at}[{i}]")
            if d:
                return d
        return None
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return None
        if math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            return None
        return f"{at}: {a!r} -> {b!r}"
    return None if a == b else f"{at}: {a!r} -> {b!r}"


def compare_file(
    ref: Path, cand: Path, rtol: float = RTOL, atol: float = ATOL
) -> tuple[str, str]:
    if ref.read_bytes() == cand.read_bytes():
        return "identical", ""
    if ref.suffix == ".csv":
        ok, detail = _compare_csv(ref, cand, rtol, atol)
    elif ref.suffix == ".json":
        diff = _walk(
            json.loads(ref.read_text()), json.loads(cand.read_text()), rtol, atol
        )
        ok, detail = diff is None, diff or ""
    else:
        ok, detail = False, "binary content differs"
    return ("close" if ok else "differs"), detail


def compare_dirs(
    reference: Path, candidate: Path, rtol: float = RTOL, atol: float = ATOL
) -> list[FileResult]:
    ref_files, cand_files = _files(reference), _files(candidate)
    out = []
    for name in sorted(ref_files | cand_files):
        if name not in cand_files:
            out.append(FileResult(name, "not regenerated"))
        elif name not in ref_files:
            out.append(FileResult(name, "new"))
        else:
            status, detail = compare_file(
                reference / name, candidate / name, rtol, atol
            )
            out.append(FileResult(name, status, detail))
    return out


def summary(results: list[FileResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    order = ("identical", "close", "differs", "not regenerated", "new")
    return ", ".join(f"{counts[s]} {s}" for s in order if s in counts)
