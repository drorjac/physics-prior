"""Where the tensors live, and the dtype that decision forces.

This package has always run on the CPU, in **float64**, and that pairing is
deliberate rather than accidental: the physics terms differentiate the
network twice, and a second derivative in float32 loses about half the digits
a first derivative keeps. On the 1-D tracks the CPU is also simply fast
enough -- a PINN fit is ~25 s and the bottleneck is the optimiser's serial
epochs, not the matrix work.

That stops being true for a field. A 2-D residual over 10^4 collocation
points with second derivatives in both coordinates is where a GPU earns its
place, so the device is now selectable.

    PHYSPRIOR_DEVICE=cuda      an NVIDIA GPU
    PHYSPRIOR_DEVICE=mps       Apple Silicon
    PHYSPRIOR_DEVICE=cpu       force the CPU (the default remains automatic)
    PHYSPRIOR_DTYPE=float32    trade precision for speed, deliberately

**The GPU paths are written but NOT verified**: the machine this was
developed on reports `cuda.is_available() = False` and
`backends.mps.is_available() = False`, so every number in this repository was
produced on the CPU in float64. Treat the device selection as untested code
until someone runs it on hardware that has one.

One consequence is not a preference but an arithmetic fact: **MPS does not
support float64 at all.** Selecting it silently demotes the whole package to
float32, which is why `resolve()` says so out loud rather than letting a
second-derivative residual quietly lose half its precision.
"""

from __future__ import annotations

import os
import warnings

import torch


def available() -> dict:
    """What this machine actually offers. Used by `physprior info`."""
    return {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(torch.backends.mps.is_available()),
        "cpu": True,
        "threads": int(torch.get_num_threads()),
    }


def resolve(requested: str | None = None) -> tuple[torch.device, torch.dtype]:
    """Pick a device and the dtype it permits.

    Returns `(device, dtype)` together because the two are not independent:
    asking for MPS is asking for float32, whether or not you meant to.
    """
    name = (requested or os.environ.get("PHYSPRIOR_DEVICE", "auto")).lower()
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn("cuda requested but not available; using the cpu", stacklevel=2)
        name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        warnings.warn("mps requested but not available; using the cpu", stacklevel=2)
        name = "cpu"

    wanted = os.environ.get("PHYSPRIOR_DTYPE", "").lower()
    if name == "mps":
        # Not a policy choice. MPS has no float64 kernels.
        if wanted == "float64":
            warnings.warn(
                "mps does not support float64; falling back to float32. The "
                "physics terms here take SECOND derivatives, which lose about "
                "half their digits at single precision -- prefer cpu or cuda "
                "when a recovered constant is the result.",
                stacklevel=2,
            )
        dtype = torch.float32
    elif wanted == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float64
    return torch.device(name), dtype


def describe() -> str:
    device, dtype = resolve()
    have = available()
    return (
        f"device={device.type} dtype={str(dtype).removeprefix('torch.')} "
        f"(cuda={have['cuda']} mps={have['mps']} threads={have['threads']})"
    )
