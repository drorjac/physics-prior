"""JPL Horizons solar-system ephemerides -- track R.

Two products, both from DE441, the ephemeris that underlies modern
solar-system dynamics and which has general relativity built into it.

`planets()`   osculating elements (semi-major axis, sidereal period) for the
              eight planets at a common epoch -- the data for Kepler's third
              law.

`vectors()`   heliocentric position and velocity of one body on a fine, even
              time grid -- the data for the acceleration-space test of the
              relativistic correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from physprior.constants import AU_M, DAY_S
from physprior.units import require, require_not_none

from ..cache import cached_get, provenance

URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SUN_CENTRE = "500@10"
# The solar-system barycentre. Only the Sun's own motion is fetched against
# it: every other barycentric state is (heliocentric state + the Sun's).
SSB_CENTRE = "500@0"

# Horizons body ids and GM in km^3/s^2 (DE441 / IAU 2015 nominal).
BODIES = {
    "Mercury": ("199", 2.2031868551e4),
    "Venus": ("299", 3.24858592000e5),
    "Earth": ("399", 3.98600435507e5),  # Earth alone
    "Mars": ("499", 4.282837362069e4),  # Mars system
    "Jupiter": ("599", 1.266865341960e8),
    "Saturn": ("699", 3.793120623810e7),
    "Uranus": ("799", 5.793951256527e6),
    "Neptune": ("899", 6.835103145462e6),
}
MOON_GM = 4.9028001184e3  # km^3/s^2, needed with body 399
SUN_GM_KM = 1.32712440018e11  # km^3/s^2


def _horizons(params: dict, filename: str) -> str:
    return cached_get(URL, filename, params=params).read_text(errors="replace")


def _block(text: str) -> list[str]:
    i, j = text.find("$$SOE"), text.find("$$EOE")
    require(
        i > 0 and j > i,
        f"Horizons: no $$SOE/$$EOE block -- server said: {text[:300]!r}",
    )
    return [ln for ln in text[i + 5 : j].strip().splitlines() if ln.strip()]


@dataclass
class PlanetTable:
    name: list[str]
    a_au: np.ndarray  # semi-major axis, AU
    period_d: np.ndarray  # sidereal period, days
    ecc: np.ndarray
    provenance: list


def planets(epoch: str = "2020-01-01") -> PlanetTable:
    names, a, per, ecc, prov = [], [], [], [], []
    for name, (cmd, _gm) in BODIES.items():
        p = {
            "format": "text",
            "COMMAND": cmd,
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "ELEMENTS",
            "CENTER": SUN_CENTRE,
            "START_TIME": epoch,
            "STOP_TIME": _plus_day(epoch),
            "STEP_SIZE": "1d",
            "REF_PLANE": "ECLIPTIC",
            "OUT_UNITS": "AU-D",
            "CSV_FORMAT": "NO",
        }
        fn = f"horizons_elements_{name.lower()}_{epoch}.txt"
        txt = _horizons(p, fn)
        rec = "\n".join(_block(txt)[:6])
        a_ = _grab(rec, "A")
        pr = _grab(rec, "PR")  # sidereal period, days
        ec = _grab(rec, "EC")
        names.append(name)
        a.append(a_)
        per.append(pr)
        ecc.append(ec)
        prov.append(
            provenance(
                cached_get(URL, fn, params=p), URL, f"DE441 elements {name} @{epoch}"
            )
        )
    a_arr, per_arr, ecc_arr = np.array(a), np.array(per), np.array(ecc)
    require(0.3 < a_arr[0] < 0.5, f"Horizons: Mercury a = {a_arr[0]} is not ~0.387 AU")
    require(29 < a_arr[-1] < 31, f"Horizons: Neptune a = {a_arr[-1]} is not ~30 AU")
    require(80 < per_arr[0] < 95, f"Horizons: Mercury P = {per_arr[0]} d, not ~88")
    return PlanetTable(names, a_arr, per_arr, ecc_arr, prov)


def _plus_day(date: str) -> str:
    import datetime as dt

    return (dt.date.fromisoformat(date) + dt.timedelta(days=2)).isoformat()


def _grab(record: str, key: str) -> float:
    m = require_not_none(
        re.search(rf"\b{key}\s*=\s*([-+0-9.eED]+)", record),
        f"Horizons: key {key} not in element record",
    )
    return float(m.group(1).replace("D", "E"))


@dataclass
class Vectors:
    jd: np.ndarray  # Julian date, TDB
    t_s: np.ndarray  # seconds from the first epoch
    r_m: np.ndarray  # (N,3) position relative to the centre, metres
    v_ms: np.ndarray  # (N,3) velocity relative to the centre, m/s
    body: str
    provenance: dict


def vectors(
    body: str, start: str, stop: str, step: str = "0.5d", centre: str = "sun"
) -> Vectors:
    """`body` is a key of BODIES, or a raw Horizons id.

    Raw ids matter: "3" is the Earth-Moon BARYCENTRE and "399" is the Earth
    alone. The perturbation model pairs each body with a system GM, so it must
    ask for the barycentre, not the planet.

    `centre` is "sun" (heliocentric, the default) or "ssb" (the solar-system
    barycentre, which the relativistic n-body equations are written in).
    """
    require(centre in ("sun", "ssb"), f"Horizons: unknown centre {centre!r}")
    cmd = BODIES[body][0] if body in BODIES else str(body)
    p = {
        "format": "text",
        "COMMAND": cmd,
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": SUN_CENTRE if centre == "sun" else SSB_CENTRE,
        "START_TIME": start,
        "STOP_TIME": stop,
        "STEP_SIZE": step,
        "VEC_TABLE": "2",
        "REF_PLANE": "ECLIPTIC",
        "OUT_UNITS": "AU-D",
        "CSV_FORMAT": "YES",
    }
    # The heliocentric name is unchanged, so every existing cache still hits.
    tag = "" if centre == "sun" else "ssb_"
    fn = f"horizons_vec_{tag}{body.lower()}_{start}_{stop}_{step}.txt"
    txt = _horizons(p, fn)
    rows = []
    for ln in _block(txt):
        f = [c.strip() for c in ln.split(",")]
        if len(f) < 8:
            continue
        rows.append([float(f[0])] + [float(v) for v in f[2:8]])
    arr = np.asarray(rows)
    require(len(arr) > 10, f"Horizons: only {len(arr)} vector rows for {body}")
    jd = arr[:, 0]
    r = arr[:, 1:4] * AU_M
    v = arr[:, 4:7] * AU_M / DAY_S
    # Even grid: the finite-difference derivatives below assume it.
    d = np.diff(jd)
    require(bool(np.ptp(d) < 1e-6 * np.mean(d)), "Horizons: time grid is not even")
    rr = np.linalg.norm(r, axis=1) / AU_M
    # About the barycentre the Sun wanders within ~2 solar radii (0.01 AU).
    lo, hi = (0.05, 60.0) if centre == "sun" else (0.0, 60.0)
    require(
        bool(rr.min() > lo and rr.max() < hi),
        f"Horizons: {body} r = {rr.min():.3f}-{rr.max():.3f} AU is not sane",
    )
    return Vectors(
        jd=jd,
        t_s=(jd - jd[0]) * DAY_S,
        r_m=r,
        v_ms=v,
        body=body,
        provenance=provenance(
            cached_get(URL, fn, params=p),
            URL,
            f"DE441 vectors {body} {start}..{stop} {step}"
            + ("" if centre == "sun" else " about the barycentre"),
        ),
    )
