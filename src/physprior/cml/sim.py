"""Simulated links where the law is exactly what was put in.

Each link takes a real (frequency, length) pair from OpenMRG and gets its own
rain record at 1 min:

- wet and dry hours from a two-state Markov chain with stationary wet
  fraction 7 % and mean wet spell `wet_hours`;
- inside wet hours, rain r_t = w_t (i.i.d.) or r_t = phi r_{t-1} + (1 - phi)
  w_t (AR(1)), w_t ~ Gamma(shape, scale);
- attenuation A_t = Q( A0 + d_t + k r_t^alpha L + W_t + e_t ), with A0 a
  per-link level, d_t a diurnal drift of 0.2 dB amplitude, k and alpha from
  ITU-R P.838-3, W_t wet-antenna attenuation (optional), e_t ~ N(0, sigma^2)
  and Q rounding to 0.3 dB.

Wet-antenna attenuation follows a first-order response to rain: it relaxes
towards 2.3 (1 - exp(-r / 1 mm/h)) dB with a 5-min time constant while it
rains and decays with a 60-min time constant after (the exponential rise and
decay of Schleiss et al. 2013, capped at 2.3 dB as in Jacoby et al. 2026).

The Gamma and AR parameters are fitted to the OpenMRG gauge record when it
has been built (`fit_rain`), otherwise defaults of the same order are used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .itu import k_alpha

WET_FRACTION = 0.07
QUANT_DB = 0.3


@dataclass(frozen=True)
class SimConfig:
    name: str = "sim_ar1_waa"
    process: str = "ar1"  # "iid" | "ar1"
    waa: bool = True
    sigma_db: float = 0.15
    n_links: int = 20
    days: int = 60
    wet_hours: float = 3.0
    shape: float = 0.6
    scale: float = 4.0
    phi: float = 0.9
    waa_max_db: float = 2.3
    seed: int = 0

    def with_noise(self, sigma: float) -> SimConfig:
        return replace(self, sigma_db=sigma, name=f"{self.name}_s{sigma:g}")


SCENARIOS = {
    "sim_iid": SimConfig("sim_iid", "iid", False),
    "sim_ar1": SimConfig("sim_ar1", "ar1", False),
    "sim_iid_waa": SimConfig("sim_iid_waa", "iid", True),
    "sim_ar1_waa": SimConfig("sim_ar1_waa", "ar1", True),
}


def fit_rain(reference: pd.DataFrame) -> dict:
    """Gamma (method of moments) and lag-1 autocorrelation of the wet minutes
    of a 1-min reference record."""
    shapes, scales, phis = [], [], []
    for _, g in reference.groupby("link_id"):
        r = g.sort_values("time")["rain_mmh"].to_numpy(float)
        r = r[np.isfinite(r)]
        w = r[r > 0]
        if len(w) < 50:
            continue
        m, v = w.mean(), w.var()
        shapes.append(m * m / v)
        scales.append(v / m)
        a, b = r[:-1], r[1:]
        wet = (a > 0) & (b > 0)
        if wet.sum() > 20:
            phis.append(np.corrcoef(a[wet], b[wet])[0, 1])
    return {
        "shape": float(np.median(shapes)),
        "scale": float(np.median(scales)),
        "phi": float(np.clip(np.median(phis), 0.0, 0.99)),
    }


def simulate(cfg: SimConfig, links: pd.DataFrame) -> dict:
    """Links, series, reference and meta in the `sources.common` format."""
    rng = np.random.default_rng(cfg.seed)
    pick = links.sample(n=min(cfg.n_links, len(links)), random_state=cfg.seed)
    n = cfg.days * 24 * 60
    time = pd.date_range("2020-06-01", periods=n, freq="1min")
    p_ww = 1.0 - 1.0 / cfg.wet_hours
    p_dw = WET_FRACTION * (1 - p_ww) / (1 - WET_FRACTION)
    rows_s, rows_r, rows_l = [], [], []
    for j, (_, lk) in enumerate(pick.iterrows()):
        lid = f"s{j:02d}"
        hours = cfg.days * 24
        wet_h = np.zeros(hours, bool)
        state = rng.random() < WET_FRACTION
        for h in range(hours):
            state = rng.random() < (p_ww if state else p_dw)
            wet_h[h] = state
        wet = np.repeat(wet_h, 60)
        w = rng.gamma(cfg.shape, cfg.scale, n)
        r = np.zeros(n)
        if cfg.process == "iid":
            r = np.where(wet, w, 0.0)
        else:
            prev = 0.0
            for t in range(n):
                if wet[t]:
                    prev = cfg.phi * prev + (1 - cfg.phi) * w[t]
                    r[t] = prev
                else:
                    prev = 0.0
        k, a = k_alpha(float(lk["freq_ghz"]), str(lk["pol"]))
        L = float(lk["length_km"])
        rain_db = k * np.power(r, a) * L
        waa = np.zeros(n)
        if cfg.waa:
            target = cfg.waa_max_db * (1 - np.exp(-r / 1.0))
            v = 0.0
            for t in range(n):
                tc = 5.0 if r[t] > 0 else 60.0
                v += (target[t] - v) / tc
                waa[t] = v
        a0 = rng.uniform(40, 70)
        drift = 0.2 * np.sin(2 * np.pi * (np.arange(n) / 1440.0 + rng.random()))
        attn = a0 + drift + rain_db + waa + rng.normal(0, cfg.sigma_db, n)
        attn = np.round(attn / QUANT_DB) * QUANT_DB
        rows_s.append(pd.DataFrame({"link_id": lid, "time": time, "attn_db": attn}))
        rows_r.append(pd.DataFrame({"link_id": lid, "time": time, "rain_mmh": r}))
        rows_l.append(
            {
                "link_id": lid,
                "freq_ghz": float(lk["freq_ghz"]),
                "pol": str(lk["pol"]),
                "length_km": L,
                "lat_a": 0.0,
                "lon_a": 0.0,
                "lat_b": 0.0,
                "lon_b": 0.0,
                "ref_kind": "truth",
                "ref_ids": "",
                "ref_dist_km": 0.0,
                "attn_kind": "tsl-rsl",
                "quant_db": QUANT_DB,
            }
        )
    meta = {
        "dataset": cfg.name,
        "ref_minutes": 1,
        "start": str(time[0].date()),
        "end": str(time[-1].date()),
        "n_links": len(rows_l),
        "selection": "frequencies and lengths of OpenMRG links",
        "sources": [],
        "notes": repr(cfg),
    }
    return {
        "links": pd.DataFrame(rows_l),
        "series": pd.concat(rows_s, ignore_index=True),
        "reference": pd.concat(rows_r, ignore_index=True),
        "meta": meta,
    }
