"""Physical constants and published reference values.

Every constant carries its source. Every quantity carries its unit in the
name or the comment. Nothing in this file is fitted -- these are the numbers
the project is *judged against*, never the numbers it is given.

Sources
-------
CODATA22   CODATA 2022 recommended values (Mohr et al., Rev. Mod. Phys. 2025).
SI2019     Exact by definition of the SI base units since 2019-05-20.
IAU2015    IAU 2015 Resolution B3 nominal solar conversion constants.
FIXSEN09   Fixsen, ApJ 707, 916 (2009): T_CMB from COBE/FIRAS + WMAP.
GWTC1      Abbott et al., Phys. Rev. X 9, 031040 (2019), GWTC-1 catalogue,
           values as served by the GWOSC event API for GW150914-v3.
DE440      Park et al., AJ 161, 105 (2021), the DE440/DE441 ephemeris paper
           (Sun's radius, moment of inertia, rotation and pole, sec. 3.4),
           and the J2SUN constant in the DE440 header file,
           ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de440/header.440.
DIPOLE     Magnetic-dipole spin-down in vacuum: Pacini, Nature 219, 145 (1968);
           Gunn & Ostriker, Nature 221, 454 (1969).
MT77       Manchester & Taylor, Pulsars (Freeman, 1977): surface field for
           R = 10 km, I = 1e45 g cm^2 and an orthogonal rotator.
NISTASD    NIST Atomic Spectra Database v5.12 (Kramida, Ralchenko, Reader and
           NIST ASD Team), physics.nist.gov/asd.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Universal constants
# --------------------------------------------------------------------------
C_LIGHT = 299792458.0  # m / s                      [SI2019, exact]
G_NEWTON = 6.67430e-11  # m^3 / (kg s^2)             [CODATA22]
H_PLANCK = 6.62607015e-34  # J s                        [SI2019, exact]
HBAR = H_PLANCK / (2.0 * math.pi)  # J s
K_BOLTZMANN = 1.380649e-23  # J / K                      [SI2019, exact]

# The combination the CMB spectrum is actually sensitive to. A blackbody
# constrains h/k (through x = h nu / k T) and h (through the prefactor),
# never h and k separately without an independent temperature scale.
H_OVER_K = H_PLANCK / K_BOLTZMANN  # K s                       [derived, exact]

# --------------------------------------------------------------------------
# Atomic (track A -- NIST hydrogen levels)
# --------------------------------------------------------------------------
RYDBERG_INF_M = 10973731.568157  # 1 / m, infinite-mass      [CODATA22]
M_ELECTRON = 9.1093837139e-31  # kg                        [CODATA22]
M_PROTON = 1.67262192595e-27  # kg                        [CODATA22]

# Finite-nuclear-mass Rydberg for hydrogen-1: R_H = R_inf * m_p/(m_e + m_p).
# This -- not R_inf -- is what a fit to H I levels must return.
RYDBERG_H_M = RYDBERG_INF_M * M_PROTON / (M_ELECTRON + M_PROTON)  # 1 / m
RYDBERG_H_CM = RYDBERG_H_M / 100.0  # 1 / cm

# --------------------------------------------------------------------------
# Atomic (quantum/helium -- NIST He I levels)
# --------------------------------------------------------------------------
M_ALPHA = 6.6446573450e-27  # kg, alpha particle         [CODATA22]

# Finite-mass Rydberg for the Rydberg electron of He I. The electron orbits a
# He+ core of mass m_alpha + m_e, so the reduced mass is
# m_e (m_alpha + m_e) / (m_alpha + 2 m_e), not m_e m_alpha / (m_alpha + m_e).
RYDBERG_HE_M = RYDBERG_INF_M * (M_ALPHA + M_ELECTRON) / (M_ALPHA + 2.0 * M_ELECTRON)
RYDBERG_HE_CM = RYDBERG_HE_M / 100.0  # 1 / cm

# First ionisation energy of He I, as the series limit of the level table.
HE_I_IONISATION_CM = 198310.66637  # 1 / cm, +- 0.00020         [NISTASD]

# --------------------------------------------------------------------------
# Pulsars (gravity/pulsar_spindown -- ATNF catalogue)
# --------------------------------------------------------------------------
# nu_dot = -K nu^n for a dipole spinning in vacuum: n = 3 exactly.
BRAKING_INDEX_DIPOLE = 3.0  # dimensionless              [DIPOLE]
# B_s = 3.2e19 sqrt(P Pdot) gauss, P in s.
B_SURFACE_COEFF_G = 3.2e19  # G s^-1/2                    [MT77]
JULIAN_YEAR_S = 365.25 * 86400.0  # s, the Julian year          [exact]

# --------------------------------------------------------------------------
# Cosmological (track Q -- COBE/FIRAS)
# --------------------------------------------------------------------------
T_CMB_K = 2.72548  # K                          [FIXSEN09]
T_CMB_K_ERR = 0.00057  # K, 1 sigma                 [FIXSEN09]

# --------------------------------------------------------------------------
# Solar system (track R -- JPL Horizons)
# --------------------------------------------------------------------------
GM_SUN = 1.32712440018e20  # m^3 / s^2                  [IAU2015]
AU_M = 1.495978707e11  # m                          [IAU2015, exact]
DAY_S = 86400.0  # s                          [exact]
ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi
JULIAN_CENTURY_D = 36525.0  # days

# Einstein's 1915 result for Mercury, the number the track is aimed at.
MERCURY_GR_PRECESSION_ARCSEC_CY = 42.98  # " / century   [Einstein 1915; Will 2014]

# What DE441 puts into the Sun beyond a point mass. Used only to ask whether
# the terms the Mercury model leaves out explain its residual -- never fitted.
J2_SUN = 2.1961391516529825e-7  # dimensionless             [DE440 header]
R_SUN_DE440_M = 6.96e8  # m, equatorial radius      [DE440]
SUN_C_OVER_MR2 = 0.068842  # polar moment of inertia   [DE440]
SUN_ROTATION_DEG_PER_DAY = 14.1844  # deg / day                  [DE440]
SUN_POLE_RA_DEG = 286.13  # ICRF, J2000              [DE440; Archinal 2018]
SUN_POLE_DEC_DEG = 63.87  # ICRF, J2000              [DE440; Archinal 2018]
# 84381.448 arcsec: the obliquity Horizons uses for its "ECLIPTIC" frame.
OBLIQUITY_J2000_DEG = 23.4392911  # ecliptic vs ICRF equator  [IAU 1976]

# --------------------------------------------------------------------------
# Gravitational waves (track G -- LIGO GW150914)
# --------------------------------------------------------------------------
M_SUN_KG = GM_SUN / G_NEWTON  # kg, derived from GM_sun
# Solar mass in seconds: GM_sun / c^3. The natural unit for waveform algebra.
T_SUN_S = GM_SUN / C_LIGHT**3  # s  (4.9255e-6)

GW150914_GPS_MERGER = 1126259462.4  # s, GPS                [GWTC1]
GW150914_MCHIRP_SOURCE = 28.6  # M_sun, source frame   [GWTC1]
GW150914_REDSHIFT = 0.09  # dimensionless         [GWTC1]
# A waveform's frequency evolution fixes the REDSHIFTED (detector-frame)
# chirp mass. That is the only mass this project can recover from strain.
GW150914_MCHIRP_DETECTOR = GW150914_MCHIRP_SOURCE * (1.0 + GW150914_REDSHIFT)

# The unit guards that used to live here are in `physprior.units`, which
# raises `UnitError` rather than `AssertionError` -- assertions are stripped by
# `python -O`, and a guard that vanishes under optimisation is worse than none.

__all__ = [_n for _n in dir() if not _n.startswith("_")]
