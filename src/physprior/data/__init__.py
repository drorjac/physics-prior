"""The data layer.

One module per source under `physprior.data.sources`. Every loader

  * downloads once into `settings.raw_dir` and never re-fetches;
  * keeps the bytes exactly as served, so a result can be traced to its file;
  * records provenance (URL, byte count, SHA-256) alongside the data;
  * asserts the units and ranges it promises, raising `UnitError` if not.

Nothing in here knows about fitting, physics problems or plots.
"""

from __future__ import annotations

from .cache import Provenance, cached_get
from .registry import DATASETS, load

__all__ = ["DATASETS", "Provenance", "cached_get", "load"]
