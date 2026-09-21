"""One module per data source.

    gwosc      LIGO strain around GW150914 (Gravitational Wave Open Science Center)
    firas      the COBE/FIRAS CMB monopole spectrum (NASA LAMBDA)
    nist       hydrogen energy levels (NIST Atomic Spectra Database)
    horizons   planetary elements and state vectors (JPL Horizons, DE441)

Each downloads once, records provenance, and asserts the units and ranges it
promises. None of them knows anything about fitting.
"""

from __future__ import annotations

__all__ = ["firas", "gwosc", "horizons", "nist"]
