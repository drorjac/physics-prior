"""NIST ASD hydrogen (H I) energy levels -- track A.

The n-averaged levels E_n for n = 1..40, in cm^-1 above the ground state,
with quoted uncertainties, plus the ionisation limit. Retrieved from the NIST
Atomic Spectra Database, Kramida et al., version 5.12.

The law to be discovered is Bohr's:

    E_n - E_1 = R (1 - 1/n^2)          [cm^-1]

and it is *not exact*. The n -> infinity limit of the data is the measured
ionisation energy, 109678.771743 cm^-1, while the Bohr-with-reduced-mass
Rydberg is R_H = 109677.583 cm^-1. The 1.19 cm^-1 gap (11 ppm) is the
relativistic + QED (Lamb shift) correction to the 1s level. It is physics,
not fit error, and the track reports it as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physprior.units import require, require_not_none

from ..cache import cached_get, provenance

URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
FILENAME = "nist_h_levels.tsv"
PARAMS = {
    "de": "0",
    "spectrum": "H I",
    "units": "0",
    "format": "3",
    "output": "0",
    "page_size": "15",
    "multiplet_ordered": "0",
    "conf_out": "on",
    "term_out": "on",
    "level_out": "on",
    "unc_out": "1",
    "j_out": "on",
    "temp": "",
    "submit": "Retrieve Data",
}


@dataclass
class HydrogenLevels:
    n: np.ndarray  # principal quantum number, dimensionless
    energy_icm: np.ndarray  # level above ground state, cm^-1
    sigma_icm: np.ndarray  # quoted 1-sigma uncertainty, cm^-1
    limit_icm: float  # measured ionisation energy, cm^-1
    limit_sigma_icm: float  # its uncertainty, cm^-1
    provenance: dict

    def __len__(self) -> int:
        return len(self.n)


def _cell(s: str) -> str:
    return s.strip().strip('"').strip()


def load() -> HydrogenLevels:
    path = cached_get(URL, FILENAME, params=PARAMS)
    text = path.read_text(errors="replace")
    require(
        "Configuration" in text and "Level (cm-1)" in text,
        "NIST: response is not the tab-delimited level table",
    )

    n_list, e_list, s_list = [], [], []
    limit = limit_sigma = None
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        conf, term = _cell(parts[0]), _cell(parts[1])
        # Brackets/parentheses mark interpolated or extrapolated values; the
        # number inside them is still the tabulated level.
        lvl = _cell(parts[4]).strip("[]()")
        unc = _cell(parts[6]).strip("[]()")
        if not lvl:
            continue
        if term == "Limit":
            limit, limit_sigma = float(lvl), float(unc or "nan")
            continue
        if conf == "1s":  # the ground state, n = 1
            n_list.append(1)
        elif conf.isdigit():  # the n-averaged level
            n_list.append(int(conf))
        else:
            continue  # an l-resolved sublevel: skip
        e_list.append(float(lvl))
        s_list.append(float(unc) if unc else np.nan)

    n = np.asarray(n_list, dtype=float)
    e = np.asarray(e_list, dtype=float)
    s = np.asarray(s_list, dtype=float)
    order = np.argsort(n)
    n, e, s = n[order], e[order], s[order]

    require(len(n) >= 20, f"NIST: only {len(n)} n-levels parsed")
    require(n[0] == 1 and e[0] == 0.0, "NIST: ground state is not n=1 at 0")
    require(bool(np.all(np.diff(n) == 1)), "NIST: n is not contiguous")
    require(bool(np.all(np.diff(e) > 0)), "NIST: levels are not monotone in n")
    limit = require_not_none(limit, "NIST: no ionisation limit row")
    # cm^-1, not eV and not m^-1.
    require(
        109678.0 < limit < 109679.0,
        f"NIST: limit {limit} is not the H I ionisation energy in cm^-1",
    )
    require(e[-1] < limit, "NIST: a level sits above the ionisation limit")

    return HydrogenLevels(
        n=n,
        energy_icm=e,
        sigma_icm=s,
        limit_icm=float(limit),
        limit_sigma_icm=float(limit_sigma if limit_sigma is not None else "nan"),
        provenance=provenance(path, URL, "NIST ASD v5.12, H I levels, Kramida et al."),
    )
