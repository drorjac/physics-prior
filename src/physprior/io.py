"""Saving results. Every number the notebooks and README quote comes from here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from physprior.config import get_settings


def save_table(df: pd.DataFrame, track: str, name: str) -> Path:
    path = get_settings().results(track) / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_json(obj: Any, track: str, name: str) -> Path:
    path = get_settings().results(track) / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_default, sort_keys=True))
    return path


def load_table(track: str, name: str) -> pd.DataFrame:
    return pd.read_csv(get_settings().results_dir / track / f"{name}.csv")


def load_json(track: str, name: str) -> Any:
    return json.loads((get_settings().results_dir / track / f"{name}.json").read_text())


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
