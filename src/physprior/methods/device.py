"""Where the tensors live, and the dtype that decision forces.

This package has always run on the CPU, in **float64**. On the 1-D tracks
the CPU is simply fast enough -- a PINN fit is ~25 s and the bottleneck is
the optimiser's serial epochs, not the matrix work.

That stops being true for a field. A 2-D residual over 10^4 collocation
points with second derivatives in both coordinates is where a GPU earns its
place, so the device is now selectable.

    PHYSPRIOR_DEVICE=cuda      an NVIDIA GPU
    PHYSPRIOR_DEVICE=mps       Apple Silicon
    PHYSPRIOR_DEVICE=cpu       force the CPU (the default remains automatic)
    PHYSPRIOR_DTYPE=float32    trade precision for speed, deliberately

Measured on this machine (`benchmarks/`, Apple M2, 10 GPU cores), on the
2-D residual with second derivatives:

    collocation    cpu f64    cpu f32    mps f32    speedup
          2,000    217.7ms    122.9ms     62.2ms      3.50x
         10,000    856.3ms    354.8ms    109.2ms      7.84x
         50,000   6455.5ms   3374.3ms    955.4ms      6.76x

So the GPU is worth roughly 7x once the collocation set is large -- which is
the field case, and only the field case.

**MPS does not support float64 at all**, so selecting it demotes the package
to float32 whether or not that was intended, and `resolve()` returns the
dtype alongside the device rather than letting that happen quietly.

What that demotion costs was also measured rather than asserted, on the
ANALYTIC field where alpha = 0.05 exactly:

    cpu float64    0.050000000000
    cpu float32    0.050000000991    2.0e-08 relative
    mps float32    0.049999999585    8.3e-09 relative

**Negligible** -- and an earlier version of this file claimed otherwise,
saying a float32 second derivative "loses about half the digits". The digit
count is right and the conclusion was wrong: the worst pointwise error in
u_xx is 6e-07 relative, while the network's own error in u_xx is 60%, eight
orders of magnitude larger. Arithmetic precision is not what limits this
problem. The warning below therefore flags the demotion without claiming it
is the dominant error.

**The GPU paths in the package remain UNVERIFIED**, because the project's own
torch cannot reach the GPU on this OS: torch 2.11 requires macOS 14 and this
machine runs 13.4, so `is_built()` is True and `is_available()` is False. The
benchmarks above ran under torch 2.8 in a separate venv. Every number in
`results/` was produced on the CPU in float64. See `benchmarks/README.md`.
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
                "mps does not support float64; falling back to float32. "
                "Measured cost on this project's second-derivative estimator "
                "is ~2e-08 relative (benchmarks/device_precision.py), so this "
                "is a notice, not a reason to avoid the gpu -- but prefer cpu "
                "or cuda if you need float64 reproducibility with results/.",
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
