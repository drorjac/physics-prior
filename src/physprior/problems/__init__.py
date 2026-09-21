"""The physics problems.

One subpackage per subject. Each holds, side by side:

  * its **simulations** -- where the law is exactly what was put in, so the
    error a method makes is the method's own;
  * its **real-data benchmark tracks** -- a `Problem` plus the sweeps;
  * its **discovery experiments** -- recovering the law from the simulation,
    and calibrating the pipeline that measures the real data.

    gravity      orbits, the three-body problem, the solar system; Kepler
    relativity   Schwarzschild orbits, inspiral waveforms; GW150914, Mercury
    quantum      the Schrodinger equation; NIST hydrogen, COBE/FIRAS
"""

from __future__ import annotations

PROBLEMS: tuple[str, ...] = ("gravity", "relativity", "quantum")

__all__ = ["PROBLEMS"]
