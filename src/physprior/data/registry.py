"""A name -> loader registry, so datasets can be listed and fetched by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from physprior.exceptions import DataError

DATASETS: dict[str, tuple[str, str]] = {
    "gwosc/gw150914": ("physprior.data.sources.gwosc", "frequency_track"),
    "cobe/firas": ("physprior.data.sources.firas", "load"),
    "nist/hydrogen": ("physprior.data.sources.nist", "load"),
    "nist/helium": ("physprior.data.sources.nist", "load_helium"),
    "jpl/planets": ("physprior.data.sources.horizons", "planets"),
    "noaa/isd-stations": ("physprior.data.sources.isd", "load_station_list"),
}


def loader(name: str) -> Callable[..., Any]:
    """The callable that returns a dataset, by registry name."""
    if name not in DATASETS:
        raise DataError(
            f"unknown dataset {name!r}; known: {', '.join(sorted(DATASETS))}"
        )
    module_name, attr = DATASETS[name]
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def load(name: str, **kwargs: Any) -> Any:
    """Load a dataset by registry name."""
    return loader(name)(**kwargs)
