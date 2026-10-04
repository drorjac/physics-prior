"""LIGO GW150914 strain -- track G.

32 s of 4096 Hz calibrated strain from both detectors around the first
gravitational-wave detection, from GWOSC (Abbott et al. 2019, GWTC-1).

Two products:

`conditioned()`   whitened, band-passed, coherently combined H1+L1 strain.
                  L1 is inverted (the detectors' arms are rotated ~90 deg
                  relative to each other) and delayed by the measured 6.9 ms
                  light travel time before summing.

`frequency_track()`  a MODEL-FREE estimate of the instantaneous GW frequency:
                  sub-sample zero crossings of the conditioned strain give
                  successive half periods. Nothing about binaries enters it,
                  so fitting an inspiral law to it is not circular.

The track is short on purpose and by necessity. GW150914's matched-filter SNR
of 24 is accumulated over the whole waveform; the per-cycle SNR is a few, so a
model-free frequency can only be measured for the last handful of cycles. That
is the data the methods actually compete on -- roughly ten noisy points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy import signal

from physprior.constants import GW150914_GPS_MERGER
from physprior.units import require

from ..cache import cached_get, provenance

BASE = "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
FILES = {
    "H1": "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "L1": "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
}
# Measured H1-L1 arrival delay for GW150914 (L1 first), Abbott et al. 2016.
HL_DELAY_S = 0.0069
BAND_HZ = (35.0, 250.0)  # LIGO's sensitive band for this event
PSD_SEGMENT_S = 4.0
# Envelope SNR a cycle must clear to be kept. This was originally set to 2.0
# a priori, which turned out to be badly wrong: at 2.0 the extraction admits
# noise-induced extra zero crossings that read as 200 Hz where the true
# frequency is 40 Hz, and an injected chirp mass of 31.2 comes back far too
# low (results/relativity/threshold_calibration.csv). The value below was
# calibrated on SIMULATED signals injected into this same real detector noise
# (physprior/problems/relativity/discovery.py, `threshold_calibration`, on the
# calibration seeds), never on the real event. There 2.5 and 3.0 have biases
# of similar size and opposite sign; 3.0 has the smaller scatter. It is then
# checked on the validation seeds by `injection_test`.
SNR_THRESHOLD = 3.0
ENVELOPE_WINDOW_S = 0.02


@dataclass
class ChirpTrack:
    t_s: np.ndarray  # seconds relative to the GWOSC event GPS time
    t_peak_s: float  # envelope peak = merger, seconds rel. event
    f_hz: np.ndarray  # instantaneous GW frequency, Hz
    snr: np.ndarray  # local envelope / off-source noise rms
    strain_t: np.ndarray  # conditioned strain, time axis (s rel. event)
    strain_h: np.ndarray  # conditioned strain, dimensionless (whitened)
    fs: float  # Hz
    provenance: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t_s)


def load_detector(det: str) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Raw strain for one detector. Returns (t_gps, strain, fs, provenance)."""
    fn = FILES[det]
    path = cached_get(BASE + fn, fn)
    with h5py.File(path, "r") as fh:
        ds = fh["strain/Strain"]
        x = ds[()]
        dt = float(ds.attrs["Xspacing"])
        t0 = float(ds.attrs["Xstart"])
        detector = fh["meta"]["Detector"][()].decode()
    require(detector == det, f"GWOSC: file {fn} holds {detector}, not {det}")
    fs = 1.0 / dt
    require(abs(fs - 4096.0) < 1e-6, f"GWOSC: sampling rate {fs} Hz, expected 4096")
    require(len(x) == 131072, f"GWOSC: {len(x)} samples, expected 32 s x 4096 Hz")
    require(bool(np.isfinite(x).all()), f"GWOSC: {det} strain contains NaN")
    # Strain is dimensionless and of order 1e-18 in these files.
    require(
        1e-20 < np.std(x) < 1e-16,
        f"GWOSC: {det} strain rms {np.std(x):.2e} is not dimensionless strain",
    )
    t = t0 + np.arange(len(x)) * dt
    return t, x, fs, provenance(path, BASE + fn, f"GWOSC GW150914-v3 {det} 4 kHz 32 s")


def whiten(x: np.ndarray, fs: float) -> np.ndarray:
    """Divide by the amplitude spectral density estimated from the same 32 s."""
    win = signal.windows.tukey(len(x), alpha=0.125)
    f_psd, pxx = signal.welch(x * win, fs=fs, nperseg=int(PSD_SEGMENT_S * fs))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    psd = np.interp(freqs, f_psd, pxx)
    return np.fft.irfft(np.fft.rfft(x * win) / np.sqrt(psd / (2.0 / fs)), n=len(x))


def conditioned(
    band: tuple[float, float] = BAND_HZ, inject: dict[str, np.ndarray] | None = None
):
    """Whitened, band-passed, coherently combined H1 + L1 strain.

    `inject` adds a per-detector strain series to the RAW data before
    whitening. That is how a simulated waveform is given the real detector's
    noise: the same coloured noise, the same whitening, the same filter, the
    same extraction. Injecting into flat white noise instead quietly changes
    the answer, because whitening boosts the 100-200 Hz band where the
    detectors are most sensitive and that is precisely where the informative
    high-frequency cycles live.
    """
    prov: list = []
    cond: dict[str, np.ndarray] = {}
    fs = 0.0
    t_gps = np.empty(0)
    for det, sign in (("H1", 1.0), ("L1", -1.0)):
        t, x, fs, pv = load_detector(det)
        prov.append(pv)
        t_gps = t
        if inject is not None and det in inject:
            x = x + np.asarray(inject[det], float)
        b, a = signal.butter(4, list(band), btype="bandpass", fs=fs)
        cond[det] = signal.filtfilt(b, a, whiten(x, fs)) * sign
    nshift = round(HL_DELAY_S * fs)
    comb = cond["H1"].copy()
    comb[nshift:] += cond["L1"][:-nshift]
    comb /= np.sqrt(2.0)
    return t_gps - GW150914_GPS_MERGER, comb, fs, prov


def extract_track(
    t: np.ndarray,
    h: np.ndarray,
    fs: float,
    band: tuple[float, float] = BAND_HZ,
    snr_threshold: float = SNR_THRESHOLD,
    provenance: list | None = None,
    t0: float = 0.0,
) -> ChirpTrack:
    """The frequency-track measurement, on ANY conditioned strain.

    Factored out so that a SIMULATED waveform with a known chirp mass can be
    pushed through byte-for-byte the same extraction as the real detector
    data. Any bias this pipeline has then shows up as a difference between the
    injected mass and the recovered one, which is the only way to know whether
    the number it returns on GW150914 can be trusted.
    """
    # `t0` is the nominal event time within the record; every window below is
    # relative to it, so an INJECTED signal placed in a quiet stretch of real
    # detector data goes through exactly this code path.
    rel = t - t0
    # Off-source noise level: everything more than 1 s from the event, away
    # from the 32 s record's tapered edges.
    off = (np.abs(rel) > 1.0) & (t > t[0] + 2.0) & (t < t[-1] - 2.0)
    noise_rms = float(np.std(h[off]))
    require(noise_rms > 0, "GW: zero off-source noise -- whitening failed")

    # Envelope from the analytic signal, smoothed over ~one low-frequency cycle.
    env = np.abs(signal.hilbert(h))
    k = int(ENVELOPE_WINDOW_S * fs) | 1
    env = np.convolve(env, np.ones(k) / k, mode="same")
    snr_t = env / noise_rms

    # Sub-sample zero crossings, x8 upsampled, in a window around the event.
    w = (rel > -0.30) & (rel < 0.10)
    up = 8
    hr = signal.resample_poly(h[w], up, 1)
    tr = t[w][0] + np.arange(len(hr)) / (fs * up)
    idx = np.where(np.diff(np.sign(hr)) != 0)[0]
    zc = tr[idx] - hr[idx] * (tr[idx + 1] - tr[idx]) / (hr[idx + 1] - hr[idx])
    zc = zc - t0

    f_half = 0.5 / np.diff(zc)  # half period -> frequency
    t_half = 0.5 * (zc[1:] + zc[:-1])
    snr_half = np.interp(t_half + t0, t, snr_t)

    # The merger is where the whitened envelope peaks. This is model free --
    # no waveform is assumed -- and it is the only defensible place to stop:
    # the inspiral law describes the approach to merger, not the ringdown.
    near = np.abs(rel) < 0.1
    t_peak = float(rel[near][np.argmax(env[near])])

    keep = (
        (snr_half >= snr_threshold)
        & (f_half > band[0])
        & (f_half < band[1])
        & (t_half <= t_peak)
    )
    require(
        keep.sum() >= 5,
        f"GW: only {keep.sum()} inspiral cycles above SNR {snr_threshold}",
    )

    return ChirpTrack(
        t_s=t_half[keep],
        f_hz=f_half[keep],
        snr=snr_half[keep],
        t_peak_s=t_peak,
        strain_t=t,
        strain_h=h,
        fs=fs,
        provenance=provenance or [],
    )


def frequency_track(
    band: tuple[float, float] = BAND_HZ, snr_threshold: float = SNR_THRESHOLD
) -> ChirpTrack:
    """The real GW150914 track: condition both detectors, then extract."""
    t, h, fs, prov = conditioned(band)
    return extract_track(t, h, fs, band, snr_threshold, provenance=prov)
