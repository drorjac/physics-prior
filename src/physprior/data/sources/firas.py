"""COBE/FIRAS CMB monopole spectrum -- track Q.

The most precise blackbody spectrum ever measured: 43 frequency channels with
1-sigma uncertainties, from Table 4 of Fixsen et al., ApJ 473, 576 (1996),
served by NASA LAMBDA.

Columns as distributed
    1  frequency            cm^-1
    2  monopole spectrum    MJy / sr   (2.725 K blackbody + residual)
    3  residual             kJy / sr
    4  1-sigma uncertainty  kJy / sr
    5  modelled Galaxy      kJy / sr

This module converts to SI-friendly units and asserts them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physprior.constants import C_LIGHT
from physprior.units import require

from ..cache import cached_get, provenance

URL = (
    "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/"
    "firas_monopole_spec_v1.txt"
)
FILENAME = "firas_monopole_spec_v1.txt"


@dataclass
class FirasSpectrum:
    nu_hz: np.ndarray  # frequency, Hz
    nu_icm: np.ndarray  # frequency, cm^-1 (as distributed)
    intensity: np.ndarray  # specific intensity, MJy / sr
    sigma: np.ndarray  # 1-sigma uncertainty, MJy / sr
    galaxy: np.ndarray  # modelled Galactic foreground, MJy / sr
    provenance: dict

    def __len__(self) -> int:
        return len(self.nu_hz)


def load() -> FirasSpectrum:
    path = cached_get(URL, FILENAME)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    arr = np.asarray(rows, dtype=float)
    require(arr.shape[1] == 5, f"FIRAS: expected 5 columns, got {arr.shape[1]}")
    require(len(arr) == 43, f"FIRAS: expected 43 channels, got {len(arr)}")

    nu_icm = arr[:, 0]
    # 2.27 to 21.33 cm^-1 is 68 GHz to 640 GHz -- the FIRAS band.
    require(
        2.0 < nu_icm.min() < 3.0 and 20.0 < nu_icm.max() < 22.0,
        f"FIRAS: frequency range {nu_icm.min()}-{nu_icm.max()} cm^-1 "
        "is not the published 2.27-21.33",
    )
    nu_hz = nu_icm * 100.0 * C_LIGHT  # cm^-1 -> m^-1 -> Hz
    require(bool(6e10 < nu_hz.min() < 8e10), f"FIRAS: nu_min {nu_hz.min():.3e} Hz")

    intensity = arr[:, 1]  # MJy/sr
    sigma = arr[:, 3] / 1000.0  # kJy/sr -> MJy/sr
    galaxy = arr[:, 4] / 1000.0  # kJy/sr -> MJy/sr
    require(intensity.max() < 500.0, "FIRAS: intensity is not in MJy/sr")
    # The famous error bars: a few tens of kJy/sr, i.e. <1e-3 of the peak.
    require(
        sigma.max() / intensity.max() < 1e-2,
        "FIRAS: uncertainties too large -- unit slip in column 4",
    )

    return FirasSpectrum(
        nu_hz=nu_hz,
        nu_icm=nu_icm,
        intensity=intensity,
        sigma=sigma,
        galaxy=galaxy,
        provenance=provenance(path, URL, "Fixsen et al. 1996 ApJ 473,576 Table 4"),
    )
