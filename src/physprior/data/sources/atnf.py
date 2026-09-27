"""ATNF Pulsar Catalogue: spin frequency and its first two derivatives.

The catalogue is downloaded once as its versioned package and read from
`psrcat.db`, a flat text database: one record per pulsar, records separated
by lines of `@---`, each line `NAME  value  [uncertainty]  [reference]`. An
uncertainty is quoted in units of the last digit of the value, as in
`29.946923 5` for 29.946923 +- 0.000005.

For each pulsar the timing gives the braking index

    n_obs = nu * nu_ddot / nu_dot^2

which a magnetic dipole spinning in vacuum fixes at 3. The selection rule is
fixed in `SELECTION`, before any fit, and is applied here so that every user
of the data sees the same pulsars.

Manchester, Hobbs, Teoh & Hobbs, AJ 129, 1993 (2005); the catalogue version
is read from the file and recorded.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from physprior.constants import B_SURFACE_COEFF_G, JULIAN_YEAR_S
from physprior.units import require

from ..cache import cached_get, provenance

URL = "https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz"
FILENAME = "psrcat_pkg.tar.gz"
MEMBER = "psrcat_tar/psrcat.db"

# The rule, fixed before any fit. Each line removes a population whose spin
# derivatives measure something other than the pulsar's own torque.
SELECTION = {
    "spinning_down": "nu_dot < 0",
    "isolated": "no BINARY parameter: orbital motion biases nu_dot",
    "not_in_globular_cluster": "no GC: association: cluster acceleration biases nu_dot",
    "not_a_magnetar": "TYPE has no AXP or SGR: outbursts drive the spin-down",
    "young": "characteristic age below 1e4 yr, where nu_ddot is measurable "
    "above timing noise",
    "measured": "|nu_ddot| at 5 sigma or better",
}
AGE_MAX_YR = 1.0e4
NU_DDOT_MIN_SIGMA = 5.0


@dataclass
class Pulsars:
    frame: pd.DataFrame  # one row per selected pulsar
    catalogue_version: str
    n_catalogue: int
    n_with_nu_ddot: int
    provenance: dict

    def __len__(self) -> int:
        return len(self.frame)


def _value_and_error(tokens: list[str]) -> tuple[float, float]:
    """The value and its absolute 1-sigma error, from `value [err] [ref]`."""
    text = tokens[0].replace("D", "E")
    value = float(text)
    if len(tokens) < 2:
        return value, np.nan
    try:
        last = float(tokens[1])
    except ValueError:  # a reference code where the error would be
        return value, np.nan
    mantissa, _, exponent = text.upper().partition("E")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return value, last * 10.0 ** (int(exponent or 0) - decimals)


def parse(text: str) -> pd.DataFrame:
    """Every pulsar with nu, nu_dot and nu_ddot, before any selection."""
    rows = []
    for block in text.split("@-----"):
        rec: dict[str, list[str]] = {}
        for line in block.splitlines():
            if line.strip() and not line.startswith("#"):
                name, *rest = line.split()
                rec.setdefault(name, rest)
        if not {"PSRJ", "F0", "F1", "F2"} <= rec.keys():
            continue
        f0, f0_err = _value_and_error(rec["F0"])
        f1, f1_err = _value_and_error(rec["F1"])
        f2, f2_err = _value_and_error(rec["F2"])
        rows.append(
            {
                "psrj": rec["PSRJ"][0],
                "nu": f0,
                "nu_err": f0_err,
                "nu_dot": f1,
                "nu_dot_err": f1_err,
                "nu_ddot": f2,
                "nu_ddot_err": f2_err,
                "binary": "BINARY" in rec,
                "type": " ".join(rec.get("TYPE", [])),
                "assoc": " ".join(rec.get("ASSOC", [])),
                "ref_nu_ddot": rec["F2"][-1] if len(rec["F2"]) > 2 else "",
            }
        )
    df = pd.DataFrame(rows)
    df["age_yr"] = -df.nu / (2.0 * df.nu_dot) / JULIAN_YEAR_S
    period, period_dot = 1.0 / df.nu, -df.nu_dot / df.nu**2
    with np.errstate(invalid="ignore"):
        df["b_surface_g"] = B_SURFACE_COEFF_G * np.sqrt(period * period_dot)
    df["n_obs"] = df.nu * df.nu_ddot / df.nu_dot**2
    df["n_obs_err"] = df.nu * df.nu_ddot_err / df.nu_dot**2
    return df


def select(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (df.nu_dot < 0)
        & ~df.binary
        & ~df.assoc.str.contains("GC:", regex=False)
        & ~df.type.str.contains("AXP|SGR")
        & (df.age_yr < AGE_MAX_YR)
        & (df.nu_ddot.abs() >= NU_DDOT_MIN_SIGMA * df.nu_ddot_err)
    )
    return df[keep].sort_values("age_yr").reset_index(drop=True)


def load() -> Pulsars:
    path = cached_get(URL, FILENAME)
    with tarfile.open(path) as tar:
        member = tar.extractfile(MEMBER)
        require(member is not None, f"ATNF: {MEMBER} missing from the package")
        assert member is not None  # narrows the type; require() is the guard
        text = member.read().decode("latin-1")
    m = re.search(r"#CATALOGUE\s+(\S+)", text)
    require(m is not None, "ATNF: no #CATALOGUE version line")
    assert m is not None
    everything = parse(text)
    chosen = select(everything)

    crab = everything[everything.psrj == "J0534+2200"]
    # the Crab spins at 29.9 Hz: F0 is in Hz, not in rad/s or ms
    require(len(crab) == 1, "ATNF: the Crab pulsar is missing")
    require(29.0 < float(crab.nu.iloc[0]) < 31.0, "ATNF: F0 is not in Hz")
    require(float(crab.nu_dot.iloc[0]) < 0, "ATNF: the Crab is not spinning down")
    require(len(chosen) >= 8, f"ATNF: only {len(chosen)} pulsars pass the rule")
    require(bool((chosen.age_yr > 0).all()), "ATNF: a negative characteristic age")
    return Pulsars(
        frame=chosen,
        catalogue_version=m.group(1),
        n_catalogue=text.count("PSRJ "),
        n_with_nu_ddot=len(everything),
        provenance=provenance(
            path, URL, f"ATNF Pulsar Catalogue v{m.group(1)}, Manchester et al. 2005"
        ),
    )
