"""Cached HTTP fetch, and provenance.

Everything is downloaded once into the raw-data directory and never
re-fetched. The bytes are kept exactly as served, so any result can be traced
back to the file it came from, and `Provenance` records the URL, the size and
the SHA-256 alongside it.

Downloads go through `requests` rather than a shell tool on purpose: it uses
certifi's CA bundle, which succeeds against TLS-intercepting proxies that the
system bundle rejects -- a failure mode that otherwise looks like the remote
service being down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from physprior.config import get_settings
from physprior.exceptions import DownloadError

TIMEOUT = 180


def cached_get(
    url: str, filename: str, params: dict | None = None, force: bool = False
) -> Path:
    """GET `url` into the raw-data directory, once. Returns the local path.

    Raises `DownloadError` rather than letting a `requests` exception escape,
    so a caller can tell "the data is not available" apart from a bug. With
    `PHYSPRIOR_OFFLINE=1` a cache miss is an error instead of a fetch, which
    is what makes a hermetic test run hermetic.
    """
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / filename
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    if settings.offline:
        raise DownloadError(
            f"{filename} is not cached and PHYSPRIOR_OFFLINE=1 forbids "
            f"fetching it from {url}"
        )
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"could not fetch {url}: {exc}") from exc
    path.write_bytes(resp.content)
    if path.stat().st_size == 0:
        path.unlink()
        raise DownloadError(f"empty response from {url}")
    return path


@dataclass(frozen=True)
class Provenance:
    """Where a file came from, and proof it has not changed since.

    Recorded next to every result so a number can be traced to the bytes it
    was computed from.
    """

    file: str
    bytes: int
    sha256: str
    url: str
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def provenance(path: Path, url: str, note: str = "") -> dict:
    """Provenance for `path`, as a plain dict for JSON serialisation."""
    from physprior.io import sha256

    return Provenance(
        file=path.name,
        bytes=path.stat().st_size,
        sha256=sha256(path),
        url=url,
        note=note,
    ).as_dict()
