"""NIST ASD energy levels: H I (track A) and He I (`quantum/helium`).

H I: the n-averaged levels E_n for n = 1..40, in cm^-1 above the ground
state, with quoted uncertainties, plus the ionisation limit. The law to be
discovered is Bohr's:

    E_n - E_1 = R (1 - 1/n^2)          [cm^-1]

and it is *not exact*. The n -> infinity limit of the data is the measured
ionisation energy, 109678.771743 cm^-1, while the Bohr-with-reduced-mass
Rydberg is R_H = 109677.583 cm^-1. The 1.19 cm^-1 gap (11 ppm) is the
relativistic + QED (Lamb shift) correction to the 1s level. It is physics,
not fit error, and the track reports it as such.

He I: the singly excited terms 1s.nl, n = 2..35, l = 0..7, singlet and
triplet, each the (2J+1)-weighted mean of its fine-structure levels, plus the
ionisation limit. Doubly excited states lie above the limit and are dropped.

Every level in both tables is printed in square brackets. The ASD legend
defines these as "energies determined by interpolation, extrapolation, or
other semi-empirical procedure relying on some known experimental values":
they are evaluated reference values, not raw line measurements.

Retrieved from the NIST Atomic Spectra Database v5.12, Kramida et al.
"""

from __future__ import annotations

import re
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


# --------------------------------------------------------------------------
# He I
# --------------------------------------------------------------------------

HE_FILENAME = "nist_he_levels.tsv"
HE_PARAMS = dict(PARAMS, spectrum="He I")
L_LETTERS = "spdfghik"  # spectroscopic l = 0..7 (j is skipped by convention)


@dataclass
class HeliumTerms:
    n: np.ndarray  # principal quantum number of the excited electron
    l: np.ndarray  # its orbital angular momentum, 0..7
    s: np.ndarray  # total spin: 0 singlet, 1 triplet
    energy_icm: np.ndarray  # term energy above the 1s2 ground state, cm^-1
    sigma_icm: np.ndarray  # largest quoted uncertainty among its J levels
    limit_icm: float  # ionisation energy, cm^-1
    limit_sigma_icm: float
    provenance: dict

    def __len__(self) -> int:
        return len(self.n)


def _j_weight(j: str) -> float | None:
    """2J+1 for a single J, None for an unresolved list such as "0,1,2"."""
    if "," in j or not j:
        return None
    num, _, den = j.partition("/")
    return 2.0 * (float(num) / float(den or 1)) + 1.0


def load_helium() -> HeliumTerms:
    path = cached_get(URL, HE_FILENAME, params=HE_PARAMS)
    text = path.read_text(errors="replace")
    require(
        "Configuration" in text and "Level (cm-1)" in text,
        "NIST: response is not the tab-delimited He I level table",
    )

    # (n, l, s) -> list of (energy, weight, sigma)
    terms: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    limit = limit_sigma = None
    for line in text.splitlines()[1:]:
        parts = [_cell(c) for c in line.split("\t")]
        if len(parts) < 7 or not parts[4]:
            continue
        conf, term, j = parts[0], parts[1], parts[2]
        lvl, unc = parts[4].strip("[]()"), parts[6].strip("[]()")
        if term == "Limit":
            limit, limit_sigma = float(lvl), float(unc or "nan")
            continue
        m = re.fullmatch(r"1s\.(\d+)([a-z])", conf)
        if not m or m[2] not in L_LETTERS or term[:1] not in ("1", "3"):
            continue  # ground state, doubly excited, or unparsed
        key = (int(m[1]), L_LETTERS.index(m[2]), 0 if term[0] == "1" else 1)
        weight = _j_weight(j) or 1.0
        terms.setdefault(key, []).append(
            (float(lvl), weight, float(unc) if unc else np.nan)
        )

    keys = sorted(terms)
    n = np.array([k[0] for k in keys], float)
    l = np.array([k[1] for k in keys], float)
    s = np.array([k[2] for k in keys], float)
    e = np.array(
        [
            np.average([v[0] for v in terms[k]], weights=[v[1] for v in terms[k]])
            for k in keys
        ]
    )
    sig = np.array([np.nanmax([v[2] for v in terms[k]]) for k in keys])

    limit = require_not_none(limit, "NIST: no He I ionisation limit row")
    # cm^-1: He I's first ionisation energy is 24.587 eV = 198310.7 cm^-1.
    require(
        198310.0 < limit < 198311.0,
        f"NIST: limit {limit} is not the He I ionisation energy in cm^-1",
    )
    require(len(n) >= 300, f"NIST: only {len(n)} He I terms parsed")
    require(bool(np.all(n > l)), "NIST: a term has l >= n")
    require(bool(np.all(e < limit)), "NIST: a 1s.nl term sits above the limit")
    # the lowest excited term is 1s2s 3S at 159856 cm^-1 (19.82 eV)
    require(
        159855.0 < e.min() < 159857.0,
        f"NIST: lowest excited term {e.min()} is not 1s2s 3S",
    )
    for series in {(int(a), int(b)) for a, b in zip(l, s, strict=True)}:
        sel = (l == series[0]) & (s == series[1])
        order = np.argsort(n[sel])
        require(
            bool(np.all(np.diff(e[sel][order]) > 0)),
            f"NIST: He I series l={series[0]} s={series[1]} is not monotone in n",
        )

    return HeliumTerms(
        n=n,
        l=l,
        s=s,
        energy_icm=e,
        sigma_icm=sig,
        limit_icm=float(limit),
        limit_sigma_icm=float(limit_sigma if limit_sigma is not None else "nan"),
        provenance=provenance(path, URL, "NIST ASD v5.12, He I levels, Kramida et al."),
    )
