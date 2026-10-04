"""The main comparison broken down by rain intensity, by detection and by event.

Reads the test predictions that `study.run_job` saves for every main-stage
job (seeds 11, 23, 42, pooled), and writes to results/cml/:

    intensity.csv   per dataset, arm and reference-intensity class: bins,
                    mean reference, mean estimate, relative bias, RMSE
    detection.csv   per dataset and arm: probability of detection of wet
                    bins, false-alarm ratio, and the fraction of dry bins
                    given rain, all at 0.1 mm/h
    events.csv      per dataset, arm and event-size class: events, median
                    and spread of the relative error in event total, and
                    the relative error in peak intensity

Intensity classes (mm/h, reference): dry < 0.1, light 0.1-1, moderate 1-5,
heavy 5-20, very heavy > 20. An event is a run of wet reference bins on one
link with gaps of at most 60 min; its total is the reference (or estimated)
rain summed over the bins from its first to its last wet bin. Event classes
(mm): < 1, 1-5, 5-20, > 20.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from physprior.config import get_settings

from .data import WET_MMH
from .study import _cache, jobs

CLASSES = (
    (0.0, 0.1, "dry"),
    (0.1, 1.0, "light"),
    (1.0, 5.0, "moderate"),
    (5.0, 20.0, "heavy"),
    (20.0, np.inf, "very heavy"),
)
EVENT_CLASSES = (
    (0.0, 1.0, "< 1 mm"),
    (1.0, 5.0, "1-5 mm"),
    (5.0, 20.0, "5-20 mm"),
    (20.0, np.inf, "> 20 mm"),
)
GAP_MIN = 60


def load_predictions() -> dict[str, dict]:
    """dataset -> {r, link, time, pred: {arm: array}, gate: {arm: array}},
    the three report seeds concatenated (links renumbered per seed)."""
    out: dict[str, dict] = {}
    for j in jobs("main"):
        p = _cache() / f"{j.key()}_pred.npz"
        if not p.exists():
            continue
        z = np.load(p)
        d = out.setdefault(
            j.dataset,
            {"r": [], "link": [], "time": [], "pred": {}, "gate": {}, "seed": []},
        )
        n = len(z["r"])
        d["r"].append(z["r"])
        d["link"].append(z["link"])
        d["time"].append(z["time"])
        d["seed"].append(np.full(n, j.seed))
        for k in z.files:
            if k.startswith("pred_"):
                d["pred"].setdefault(k[5:], []).append(z[k])
            if k.startswith("gate_"):
                d["gate"].setdefault(k[5:], []).append(z[k])
    for d in out.values():
        for k in ("r", "link", "time", "seed"):
            d[k] = np.concatenate(d[k])
        d["pred"] = {a: np.concatenate(v) for a, v in d["pred"].items()}
        d["gate"] = {a: np.concatenate(v) for a, v in d["gate"].items()}
    return out


def intensity_table(P: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds, d in P.items():
        r = d["r"]
        for lo, hi, name in CLASSES:
            m = (r >= lo) & (r < hi)
            if not m.any():
                continue
            for arm, p in d["pred"].items():
                e = p[m] - r[m]
                rows.append(
                    {
                        "dataset": ds,
                        "class": name,
                        "arm": arm,
                        "bins": int(m.sum()),
                        "mean_ref": float(r[m].mean()),
                        "mean_est": float(p[m].mean()),
                        "rel_bias": float(e.mean() / r[m].mean())
                        if r[m].mean() > 0
                        else np.nan,
                        "rmse": float(np.sqrt(np.mean(e**2))),
                        "gate": float(d["gate"][arm][m].mean())
                        if arm in d["gate"]
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def detection_table(P: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds, d in P.items():
        wet = d["r"] > WET_MMH
        for arm, p in d["pred"].items():
            said = p > WET_MMH
            hits = int((said & wet).sum())
            rows.append(
                {
                    "dataset": ds,
                    "arm": arm,
                    "pod": hits / max(int(wet.sum()), 1),
                    "far": int((said & ~wet).sum()) / max(int(said.sum()), 1),
                    "dry_called_wet": int((said & ~wet).sum())
                    / max(int((~wet).sum()), 1),
                }
            )
    return pd.DataFrame(rows)


def events(d: dict) -> pd.DataFrame:
    """One row per (seed, link, event): reference and estimated totals and
    peaks for every arm."""
    rows = []
    df = pd.DataFrame(
        {"seed": d["seed"], "link": d["link"], "t": d["time"], "r": d["r"]}
    )
    for a, p in d["pred"].items():
        df[a] = p
    for (_, _), g in df.groupby(["seed", "link"], sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        if len(t) < 2:
            continue
        step_h = float(np.median(np.diff(t))) / 60.0
        wet_idx = np.flatnonzero(g["r"].to_numpy() > WET_MMH)
        if len(wet_idx) == 0:
            continue
        # split wet bins into events where the gap exceeds GAP_MIN
        breaks = np.flatnonzero(np.diff(t[wet_idx]) > GAP_MIN) + 1
        for chunk in np.split(wet_idx, breaks):
            sl = slice(chunk[0], chunk[-1] + 1)
            row = {
                "total_ref": float(g["r"].iloc[sl].sum() * step_h),
                "peak_ref": float(g["r"].iloc[sl].max()),
            }
            for a in d["pred"]:
                row[f"total_{a}"] = float(g[a].iloc[sl].sum() * step_h)
                row[f"peak_{a}"] = float(g[a].iloc[sl].max())
            rows.append(row)
    return pd.DataFrame(rows)


def event_table(P: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds, d in P.items():
        ev = events(d)
        if ev.empty:
            continue
        for lo, hi, name in EVENT_CLASSES:
            m = (ev["total_ref"] >= lo) & (ev["total_ref"] < hi)
            if not m.any():
                continue
            for a in d["pred"]:
                rel = ev.loc[m, f"total_{a}"] / ev.loc[m, "total_ref"] - 1
                relp = ev.loc[m, f"peak_{a}"] / ev.loc[m, "peak_ref"] - 1
                rows.append(
                    {
                        "dataset": ds,
                        "event_class": name,
                        "arm": a,
                        "events": int(m.sum()),
                        "total_err_median": float(rel.median()),
                        "total_abs_err_median": float(rel.abs().median()),
                        "peak_err_median": float(relp.median()),
                    }
                )
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    P = load_predictions()
    out = get_settings().results("cml")
    tabs = {
        "intensity": intensity_table(P),
        "detection": detection_table(P),
        "events": event_table(P),
    }
    for k, v in tabs.items():
        v.to_csv(out / f"{k}.csv", index=False)
    (out / "analysis_meta.json").write_text(
        json.dumps({"datasets": sorted(P), "gap_min": GAP_MIN}, indent=2) + "\n"
    )
    return tabs
