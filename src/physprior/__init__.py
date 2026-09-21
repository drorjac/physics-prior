"""physprior -- what does a physics prior buy you?

A controlled comparison of physics-informed machine learning (PINNs and
symbolic regression) against a purely data-driven neural network, on real
measured data from LIGO, NASA, NIST and JPL and on simulations where the
answer is known exactly.

The package is organised along the two axes the question needs:

    physprior.data       the data layer -- one module per source, each
                         downloading once, caching, and asserting its units
    physprior.problems   the physics problems -- gravity, relativity, quantum;
                         each holds its simulations, its real-data benchmark
                         tracks and its law-discovery experiments

with `methods` (the arms), `numerics` (integrators and stencils), `benchmark`
(the protocol), `viz` and `reporting` supporting both.
"""

from __future__ import annotations

import os as _os
import warnings as _warnings
from contextlib import suppress as _suppress

__version__ = "0.1.0"
__all__ = ["PhysPriorError", "__version__", "get_settings"]

# PySR's Julia runtime must be initialised BEFORE torch or the process can
# segfault (pytorch/pytorch#78829). Both are used throughout, so the order is
# fixed here, once, rather than in each entry point. Set PHYSPRIOR_NO_JULIA=1
# to skip it when only the data layer is wanted.
if _os.environ.get("PHYSPRIOR_NO_JULIA") != "1":
    # Must be set before juliacall is imported, or Ctrl-C can segfault.
    _os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    # Having pre-imported juliacall, PySR later warns that it could not
    # configure the Julia runtime itself. That is the intended trade: the
    # import order is what prevents the segfault, and the signal handling it
    # asks about is set above. Silenced here so the warning does not appear on
    # every command; set PHYSPRIOR_NO_JULIA=1 to skip the bootstrap entirely.
    _warnings.filterwarnings(
        "ignore",
        message=".*juliacall module already imported.*",
        category=UserWarning,
    )
    with _suppress(Exception):  # PySR is an optional extra
        import juliacall as _juliacall  # noqa: F401

from .config import get_settings
from .exceptions import PhysPriorError
