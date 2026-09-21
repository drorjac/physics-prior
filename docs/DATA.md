# Data provenance

Everything is downloaded once into `data/raw/` and never edited. Each track's
`results/<track>/meta.json` records the URL, byte count and SHA-256 of every
file it read, so any number can be traced back to the bytes it came from.

| Track | File | Source | Reference |
|---|---|---|---|
| G | `H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5`, `L-L1_…` | [GWOSC](https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/) | Abbott et al., *Phys. Rev. X* **9**, 031040 (2019) — GWTC-1 |
| Q | `firas_monopole_spec_v1.txt` | [NASA LAMBDA](https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/) | Fixsen et al., *ApJ* **473**, 576 (1996), Table 4 |
| A | `nist_h_levels.tsv` | [NIST ASD](https://physics.nist.gov/asd) `energy1.pl`, H I | Kramida et al., NIST ASD v5.12 |
| R | `horizons_elements_*.txt`, `horizons_vec_*.txt` | [JPL Horizons API](https://ssd.jpl.nasa.gov/api/horizons.api) | DE441 |

## Units, and why they are asserted

The project simultaneously handles cm⁻¹ and m⁻¹, MJy/sr and W m⁻² sr⁻¹ Hz⁻¹,
AU/day and m/s, GPS seconds and seconds-before-merger, and solar masses and
seconds. A silent unit slip in any of them would be indistinguishable from a
discovery, so every loader asserts the range it promises — for example that
the FIRAS peak is ~384 MJy/sr, that the H I ionisation limit is between
109678 and 109679 cm⁻¹, and that Mercury's semi-major axis is ~0.387 AU.

## A note on `curl` and TLS

`ssd.jpl.nasa.gov` is served through an intercepting TLS proxy on this machine
and `curl`'s system CA bundle rejects it. `requests` (certifi) accepts it, so
all fetching goes through `physprior/data/cache.py`, not the shell.

## Known caveats, stated up front

- **FIRAS.** The distributed monopole spectrum is constructed as *a 2.725 K
  blackbody plus the measured residual*. Recovering T = 2.725 K is therefore
  partly by construction and is not scored as a discovery. The Planckian
  *shape*, the SR law-recovery result and the extrapolation behaviour are not
  affected by this.
- **GW150914.** The frequency track is model-free but short: seven cycles.
  That is a physical limit, not a choice — the event's SNR of 24 is
  accumulated coherently over the waveform, and a per-cycle frequency needs
  per-cycle SNR.
- **Kepler / GM_sun.** `P = 2π√(a³/GM)` ignores the planet's own mass and uses
  the osculating semi-major axis. Both bias the recovered `GM_sun` at the tens
  of ppm level; the giant planets dominate the effect.
- **Mercury / GR.** The perturbation model includes the eight planets and not
  the asteroids or the solar quadrupole. Both are below 10⁻¹⁰ of the total
  acceleration, i.e. far below the 8×10⁻⁸ GR term, and so are neglected
  deliberately rather than forgotten.
