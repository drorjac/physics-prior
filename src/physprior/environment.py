"""What produced a run: the code, the interpreter, the libraries, the device.

`physprior run` and `physprior neglected` write this to
`results/environment.json`, so every committed number can be traced to the
code and software versions that produced it. `physprior verify` ignores the
file, since it is expected to differ between machines.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from physprior import __version__
from physprior.config import get_settings
from physprior.io import sha256

# The packages whose version can move a number in results/.
PACKAGES = ("numpy", "scipy", "pandas", "sympy", "h5py", "torch", "pysr")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=get_settings().root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _torch_device() -> str | None:
    try:
        from physprior.methods import device
    except ImportError:  # torch is an optional extra
        return None
    return device.describe()


def snapshot() -> dict[str, Any]:
    root = get_settings().root
    lock = root / "tools" / "requirements.lock"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "physprior": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        # uncommitted changes to tracked files mean the commit alone does not
        # describe the code that ran
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "torch_device": _torch_device(),
        "requirements_lock_sha256": sha256(lock) if lock.is_file() else None,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
