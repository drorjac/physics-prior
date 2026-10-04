"""Controlled experiments for the relativity problem.

`injection_test`  The one that matters. A waveform with a KNOWN chirp mass is
                  simulated, buried in noise at the same per-cycle SNR as the
                  real event, and pushed through byte-for-byte the same
                  frequency-extraction and fitting pipeline that produced the
                  GW150914 number. Whatever bias comes back is the pipeline's,
                  not the universe's. Without this the real result is a number
                  with no error budget.

`precession_recovery`  From a simulated Schwarzschild orbit, recover the
                  coefficient of the GR term. Ground truth is exactly 1, so
                  the residual is the method's own floor -- the controlled
                  twin of the ephemeris measurement, which got 1.000121.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import curve_fit

from physprior.constants import GW150914_MCHIRP_DETECTOR
from physprior.data.sources import gwosc as gwdata
from physprior.exceptions import UnitError
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS

from . import spacetime
from .gw150914 import ETA, make_law

# The real conditioned GW150914 strain peaks at about 8.5x the off-source
# noise rms. The injection is scaled to match, so the comparison is fair.
TARGET_PEAK_SNR = 8.5

# Injection seeds. The SNR cut is chosen on the CALIBRATION set (the tuning
# seeds plus extra noise stretches) and checked on the VALIDATION set (the
# reporting seeds plus their own extras). The two sets share no seed, so the
# injection test is not scored on the noise the cut was chosen on.
CALIBRATION_SEEDS = (*TUNE_SEEDS, 606, 707, 808, 909, 1010, 1111, 1212)
VALIDATION_SEEDS = (*REPORT_SEEDS, 101, 202, 303, 404, 505)


def _place(
    waveform: dict, t_axis: np.ndarray, t0: float, fs: float, amplitude: float
) -> np.ndarray:
    """Put a waveform into a zero series on `t_axis`, merger at `t0`."""
    out = np.zeros_like(t_axis)
    idx = np.round((t0 + waveform["t"]) * fs).astype(int) - int(round(t_axis[0] * fs))
    ok = (idx >= 0) & (idx < len(out))
    out[idx[ok]] = amplitude * waveform["h"][ok]
    return out


@lru_cache(maxsize=16)
def _inject(
    mchirp: float,
    seed: int,
    peak_snr: float = TARGET_PEAK_SNR,
    pn_order: int = 3,
    t0: float = -8.0,
):
    """A simulated chirp injected into the REAL detector noise.

    The waveform is added to the raw H1 and L1 strain at `t0`, a quiet stretch
    8 seconds before the real event, with L1 inverted and advanced by the
    measured 6.9 ms so that the project's own conditioning recombines the two
    copies coherently -- exactly as it does for the real signal. `seed` picks
    the stretch of real noise by shifting `t0`, so each trial sees a different,
    genuine noise realisation rather than a different pseudo-random draw.
    """
    wf = spacetime.inspiral_chirp(mchirp_msun=mchirp, eta=ETA, pn_order=pn_order)
    rng = np.random.default_rng(seed)
    t0 = float(t0 + rng.uniform(-4.0, 3.0))  # a different quiet stretch

    t_ref, _, fs, _ = gwdata.load_detector("H1")
    from physprior.constants import GW150914_GPS_MERGER

    t_axis = t_ref - GW150914_GPS_MERGER

    # One calibration pass: whitening is linear in the injected amplitude, so
    # a trial injection fixes the scale needed to reach `peak_snr`.
    trial = 1e-21
    inj = {
        "H1": _place(wf, t_axis, t0, fs, trial),
        "L1": _place(wf, t_axis, t0 - gwdata.HL_DELAY_S, fs, -trial),
    }
    t, x, fs, _ = gwdata.conditioned(inject=inj)
    from scipy import signal as sg

    env = np.abs(sg.hilbert(x))
    k = int(gwdata.ENVELOPE_WINDOW_S * fs) | 1
    env = np.convolve(env, np.ones(k) / k, mode="same")
    near = np.abs(t - t0) < 0.12
    off = np.abs(t - t0) > 1.0
    got = (env[near].max() - np.median(env[off])) / np.std(x[off])
    amp = trial * (peak_snr / max(got, 1e-6))

    inj = {
        "H1": _place(wf, t_axis, t0, fs, amp),
        "L1": _place(wf, t_axis, t0 - gwdata.HL_DELAY_S, fs, -amp),
    }
    t, x, fs, _ = gwdata.conditioned(inject=inj)
    return t, x, fs, t0


def injection_test(
    mchirp_true: float = GW150914_MCHIRP_DETECTOR,
    seeds=VALIDATION_SEEDS,
    pn_orders=(0, 3),
    peak_snr: float = TARGET_PEAK_SNR,
    snr_threshold: float | None = None,
) -> dict:
    """Recover a known chirp mass through the real pipeline, many times."""
    rows = []
    th = gwdata.SNR_THRESHOLD if snr_threshold is None else snr_threshold
    for seed in seeds:
        t, x, fs, t0 = _inject(mchirp_true, seed, peak_snr)
        try:
            tr = gwdata.extract_track(t, x, fs, t0=t0, snr_threshold=th)
        except UnitError:
            rows.append({"seed": seed, "n_cycles": 0, "ok": False})
            continue
        row = {
            "seed": seed,
            "n_cycles": len(tr),
            "ok": True,
            "f_min": float(tr.f_hz.min()),
            "f_max": float(tr.f_hz.max()),
        }
        for order in pn_orders:
            law = make_law(order)
            try:
                popt, pcov = curve_fit(
                    lambda xx, Mc, tc, _l=law: _l(xx, Mc, tc),
                    tr.t_s.reshape(-1, 1),
                    tr.f_hz,
                    p0=[31.0, tr.t_peak_s],
                    bounds=([5.0, -0.05], [200.0, 0.10]),
                    maxfev=80000,
                )
                row[f"Mc_pn{order}"] = float(popt[0])
                row[f"Mc_sigma_pn{order}"] = float(np.sqrt(np.diag(pcov))[0])
            except Exception:
                row[f"Mc_pn{order}"] = np.nan
                row[f"Mc_sigma_pn{order}"] = np.nan
        rows.append(row)

    ok = [r for r in rows if r.get("ok")]
    summary = {
        "mchirp_true": mchirp_true,
        "peak_snr": peak_snr,
        "snr_threshold": th,
        "n_trials": len(seeds),
        "n_usable": len(ok),
        "median_cycles": float(np.median([r["n_cycles"] for r in ok])) if ok else 0.0,
        "trials": rows,
    }
    for order in pn_orders:
        vals = np.array([r.get(f"Mc_pn{order}", np.nan) for r in ok], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            summary[f"pn{order}"] = {
                "median_Mc": float(np.median(vals)),
                "bias_Msun": float(np.median(vals) - mchirp_true),
                "bias_pct": float((np.median(vals) - mchirp_true) / mchirp_true * 100),
                "scatter_Msun": float(np.std(vals)),
                "n": int(len(vals)),
            }
    return summary


def precession_recovery(gr_boost: float = 1.0, n_orbits: int = 5) -> dict:
    """Recover the coefficient of the GR term from a simulated orbit.

    The orbit is integrated with the term switched on; the precession is
    measured; and the coefficient implied by 6 pi GM alpha / (c^2 a (1-e^2))
    is compared with the 1 that was put in.
    """
    run = spacetime.schwarzschild_orbit(
        n_orbits=n_orbits, gr_boost=gr_boost, relativistic=True
    )
    ctrl = spacetime.schwarzschild_orbit(
        n_orbits=n_orbits, gr_boost=gr_boost, relativistic=False
    )
    measured = run.precession_per_orbit - ctrl.precession_per_orbit
    analytic = run.meta["analytic_precession_per_orbit"]
    alpha = measured / analytic * gr_boost
    return {
        "gr_boost": gr_boost,
        "n_orbits": n_orbits,
        "measured_per_orbit_rad": measured,
        "analytic_per_orbit_rad": analytic,
        "alpha_recovered": float(alpha),
        "alpha_true": float(gr_boost),
        "rel_error": float(abs(alpha - gr_boost) / gr_boost),
        "numerical_control_rad": ctrl.precession_per_orbit,
    }


def real_vs_simulated_track() -> dict:
    """Lay the real GW150914 frequency track over a simulated one at the
    published chirp mass. Same axes, same pipeline, no fitting."""
    tr = gwdata.frequency_track()
    wf = spacetime.inspiral_chirp(
        mchirp_msun=GW150914_MCHIRP_DETECTOR, eta=ETA, pn_order=3
    )
    return {
        "real_t_s": tr.t_s.tolist(),
        "real_f_hz": tr.f_hz.tolist(),
        "real_t_peak_s": tr.t_peak_s,
        "sim_t_s": wf["t"].tolist(),
        "sim_f_hz": wf["f"].tolist(),
        "mchirp_msun": GW150914_MCHIRP_DETECTOR,
        "pn_order": wf["pn_order"],
    }


def threshold_calibration(
    thresholds=(2.0, 2.5, 3.0, 3.5, 4.0),
    seeds=CALIBRATION_SEEDS,
    mchirp_true: float = GW150914_MCHIRP_DETECTOR,
    pn_order: int = 3,
) -> list[dict]:
    """Choose the extraction's SNR cut on INJECTIONS, never on the real event.

    This is the project's seed discipline applied to a signal-processing
    choice: the threshold is selected where an injected, known chirp mass
    comes back unbiased, and only then is the real event re-measured with it.

    At 2.0 -- the value originally picked a priori -- the recovered mass is
    far too low, because noise adds spurious zero crossings at low envelope
    SNR. Run on CALIBRATION_SEEDS only; the adopted cut is then checked on
    VALIDATION_SEEDS by `injection_test`.
    """
    law = make_law(pn_order)
    rows = []
    for th in thresholds:
        vals, cycles, failures = [], [], 0
        for seed in seeds:
            t, x, fs, t0 = _inject(mchirp_true, seed)
            try:
                tr = gwdata.extract_track(t, x, fs, t0=t0, snr_threshold=th)
            except UnitError:
                failures += 1
                continue
            cycles.append(len(tr))
            try:
                popt, _ = curve_fit(
                    lambda xx, Mc, tc, _l=law: _l(xx, Mc, tc),
                    tr.t_s.reshape(-1, 1),
                    tr.f_hz,
                    p0=[31.0, tr.t_peak_s],
                    bounds=([5.0, -0.05], [200.0, 0.10]),
                    maxfev=80000,
                )
                vals.append(float(popt[0]))
            except Exception:
                failures += 1
        v = np.asarray(vals)
        rows.append(
            {
                "snr_threshold": th,
                "n_seeds": len(seeds),
                "n_usable": int(len(v)),
                "n_failed": failures,
                "median_cycles": float(np.median(cycles)) if cycles else float("nan"),
                "median_Mc": float(np.median(v)) if len(v) else float("nan"),
                "bias_pct": float((np.median(v) - mchirp_true) / mchirp_true * 100)
                if len(v)
                else float("nan"),
                "scatter_Msun": float(np.std(v)) if len(v) else float("nan"),
                "frac_within_20pct": float(
                    np.mean(np.abs(v - mchirp_true) / mchirp_true < 0.20)
                )
                if len(v)
                else 0.0,
            }
        )
    return rows
