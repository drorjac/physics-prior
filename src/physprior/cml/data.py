"""From a built dataset to model samples.

One sample is one target bin of one link. Its input is the excess attenuation
x = A - baseline over the `history` minutes that end with the bin; its target
is the reference rain rate averaged over the bin. The baseline is a causal
24-h rolling median of A per link, so nothing after the bin is used.

The bin length is `max(10, ref_minutes)` minutes, and the input step is the
series sampling interval (1 min for most archives). Bins with more than 20 %
of the input missing, or with no reference, are dropped.

Days are split 70/15/15 into train, validation and test, stratified by daily
rain and drawn with a fixed generator, so the split is the same for every arm
and every model seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .itu import k_alpha

BASELINE_HOURS = 24
MAX_MISSING = 0.2
WET_MMH = 0.1  # a bin is wet above this reference rate
SPLIT_SEED = 0


@dataclass
class Samples:
    """Arrays for one dataset, every sample of every link."""

    x: np.ndarray  # (n, steps) excess attenuation, dB
    m: int  # the last m steps are the target bin
    r: np.ndarray  # (n,) reference rain, mm/h
    link: np.ndarray  # (n,) link index
    day: np.ndarray  # (n,) day index
    time: np.ndarray  # (n,) bin start
    links: pd.DataFrame  # one row per link index: freq, pol, length, k, alpha
    split: np.ndarray = field(default_factory=lambda: np.empty(0, int))  # 0/1/2

    def subset(self, mask: np.ndarray) -> Samples:
        return Samples(
            self.x[mask],
            self.m,
            self.r[mask],
            self.link[mask],
            self.day[mask],
            self.time[mask],
            self.links,
            self.split[mask] if self.split.size else self.split,
        )

    def __len__(self) -> int:
        return len(self.r)


def make_samples(
    links: pd.DataFrame,
    series: pd.DataFrame,
    reference: pd.DataFrame,
    meta: dict,
    history_min: int = 60,
) -> Samples:
    step = int(meta.get("series_minutes", 1))
    bin_min = max(10, int(meta["ref_minutes"]))
    bin_min = int(np.ceil(bin_min / step) * step)
    history_min = max(history_min, bin_min)
    n_steps = history_min // step
    m = bin_min // step

    links = links.reset_index(drop=True).copy()
    k, a = zip(
        *(k_alpha(f, p) for f, p in zip(links["freq_ghz"], links["pol"], strict=True)),
        strict=True,
    )
    links["k_itu"] = np.asarray(k, float)
    links["alpha_itu"] = np.asarray(a, float)

    sgrp = dict(tuple(series.groupby("link_id", sort=False)))
    rgrp = dict(tuple(reference.groupby("link_id", sort=False)))
    xs, rs, ls, ts = [], [], [], []
    for i, lid in enumerate(links["link_id"]):
        if lid not in sgrp or lid not in rgrp:
            continue
        s = sgrp[lid].set_index("time")["attn_db"].sort_index()
        s = s[~s.index.duplicated()]
        s = s.resample(f"{step}min").mean()
        # short gaps are interpolated, long ones stay missing
        s = s.interpolate(limit=max(1, 5 // step), limit_area="inside")
        base = s.rolling(f"{BASELINE_HOURS}h", min_periods=max(2, 60 // step)).median()
        x = (s - base).to_numpy(float)
        r = rgrp[lid].set_index("time")["rain_mmh"].sort_index()
        r = r[~r.index.duplicated()]
        rb = r.resample(f"{bin_min}min").mean()
        t0 = s.index[0]
        # bin b covers steps [i_end - m, i_end); the input is [i_end - n, i_end)
        bins = rb.index[(rb.index >= t0 + pd.Timedelta(minutes=history_min - bin_min))]
        i_end = ((bins - t0) / pd.Timedelta(minutes=step)).astype(int) + m
        ok = i_end <= len(x)
        bins, i_end = bins[ok], np.asarray(i_end[ok])
        if len(bins) == 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(x, n_steps)[i_end - n_steps]
        miss = np.isnan(win).mean(axis=1)
        tgt = rb.loc[bins].to_numpy(float)
        keep = (miss <= MAX_MISSING) & np.isfinite(tgt)
        win = np.nan_to_num(win[keep], nan=0.0)
        xs.append(win.astype(np.float32))
        rs.append(tgt[keep].astype(np.float32))
        ls.append(np.full(int(keep.sum()), i, np.int64))
        ts.append(np.asarray(bins[keep]))
    x = np.concatenate(xs)
    r = np.concatenate(rs)
    link = np.concatenate(ls)
    time = np.concatenate(ts)
    day = pd.DatetimeIndex(time).normalize()
    _, day_idx = np.unique(day, return_inverse=True)
    smp = Samples(x, m, r, link, day_idx.astype(np.int64), time, links)
    smp.split = day_split(smp)
    return smp


def day_split(s: Samples, frac=(0.70, 0.15, 0.15)) -> np.ndarray:
    """0 train, 1 validation, 2 test, by day, stratified by daily mean rain:
    days are ranked by rain and dealt out in blocks of 20 in the 14/3/3
    proportion, in a shuffled order within each block."""
    n_days = int(s.day.max()) + 1
    daily = np.bincount(s.day, weights=s.r, minlength=n_days) / np.maximum(
        np.bincount(s.day, minlength=n_days), 1
    )
    order = np.argsort(-daily, kind="stable")
    rng = np.random.default_rng(SPLIT_SEED)
    pattern = np.array([0] * 14 + [1] * 3 + [2] * 3)
    lab = np.empty(n_days, int)
    for b in range(0, n_days, 20):
        blk = order[b : b + 20]
        lab[blk] = rng.permutation(pattern)[: len(blk)]
    return lab[s.day]


def budget_mask(s: Samples, frac: float, seed: int = 0) -> np.ndarray:
    """Training samples restricted to a fraction of the training days, the
    same days for every arm (wet days kept in proportion)."""
    if frac >= 1.0:
        return s.split == 0
    train_days = np.unique(s.day[s.split == 0])
    daily = np.array([s.r[s.day == d].mean() for d in train_days])
    order = train_days[np.argsort(-daily, kind="stable")]
    rng = np.random.default_rng(1000 + seed)
    n_keep = max(2, round(frac * len(order)))
    # every k-th day in rain order, from a random offset: wet and dry kept
    stride = len(order) / n_keep
    off = rng.random() * stride
    pick = order[
        np.minimum((off + stride * np.arange(n_keep)).astype(int), len(order) - 1)
    ]
    return (s.split == 0) & np.isin(s.day, pick)
