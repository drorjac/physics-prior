"""The experiments: every dataset, every arm, report seeds, data budgets, noise.

Stages, each a set of jobs (dataset, seed, budget, arms):

    main     every dataset, all arms, full training set, seeds 11/23/42
    budget   real datasets and sim_ar1_waa, 5/15/40 % of the training days,
             arms pl_cal, gru and the hybrids built from them
    noise    sim_iid and sim_ar1_waa at noise sd 0.05-0.8 dB, seed 11,
             the mean gate of the phased hybrid against noise

A job's result is cached under `.cache/cml/` keyed on its full description,
so `physprior cml` resumes. Samples are built once per dataset and cached as
`.npz` beside them.

Before modelling, links whose label-free noise sd on the training days
exceeds max(0.6 dB, 1.5 quantisation steps) are dropped (a hardware or multipath problem, not rain); the
dropped ids are recorded.

Outputs in results/cml/: runs.csv (one row per job and arm), datasets.csv,
dropped_links.csv, summary.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.methods.base import REPORT_SEEDS

from .data import Samples, budget_mask, make_samples
from .sources import common
from .train import ARMS, TrainConfig, fit_arms, link_noise

log = logging.getLogger(__name__)

REAL = ("openmrg", "openrainer", "netherlands", "openmesh")
SIMS = ("sim_iid", "sim_ar1", "sim_iid_waa", "sim_ar1_waa")
BUDGETS = (0.05, 0.15, 0.40)
NOISES = (0.05, 0.1, 0.2, 0.4, 0.8)
MAX_NOISE_DB = 0.6
HISTORY_MIN = 30
VERSION = 9


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    budget: float = 1.0
    arms: tuple[str, ...] = ARMS
    noise_db: float | None = None  # simulations only: override sigma
    stage: str = "main"
    quick: bool = False

    def key(self) -> str:
        d = asdict(self)
        d.pop("stage")
        d["version"] = VERSION
        d["cfg"] = asdict(TrainConfig().quick() if self.quick else TrainConfig())
        d["history"] = HISTORY_MIN
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]


def _cache() -> Path:
    p = get_settings().cache_dir / "cml"
    p.mkdir(parents=True, exist_ok=True)
    return p


def available() -> list[str]:
    return [d for d in REAL if (common.out_dir(d) / "meta.json").exists()]


def _raw(dataset: str, noise_db: float | None) -> dict:
    if dataset.startswith("sim_"):
        from . import sim

        cfg = sim.SCENARIOS[dataset]
        base = common.load("openmrg")
        fit = sim.fit_rain(base["reference"])
        from dataclasses import replace

        cfg = replace(cfg, **fit)
        if noise_db is not None:
            cfg = cfg.with_noise(noise_db)
        return sim.simulate(cfg, base["links"])
    return common.load(dataset)


def samples(dataset: str, noise_db: float | None = None) -> tuple[Samples, list[str]]:
    """Samples after the noise check, cached; and the dropped link ids."""
    tag = f"{dataset}_n{noise_db}_h{HISTORY_MIN}_v{VERSION}"
    p = _cache() / f"samples_{tag}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        links = pd.read_json(io.StringIO(str(z["links"])))
        links["link_id"] = links["link_id"].astype(str)
        s = Samples(
            z["x"],
            int(z["m"]),
            z["r"],
            z["link"],
            z["day"],
            z["time"],
            links,
            z["split"],
        )
        return s, list(z["dropped"])
    raw = _raw(dataset, noise_db)
    s = make_samples(**raw, history_min=HISTORY_MIN)
    sd = link_noise(s, s.split == 0)
    # the limit scales with the quantisation step: on 1-dB data the robust
    # sd of a quiet link is already about 1.5 dB
    quant = s.links["quant_db"].to_numpy(float) if "quant_db" in s.links else 0.0
    limit = np.maximum(MAX_NOISE_DB, 1.5 * quant)
    if noise_db is not None:
        # the noise sweep varies the noise on purpose: no screen there
        limit = np.full_like(sd, np.inf)
    bad = np.flatnonzero(~(sd <= limit))
    dropped = [str(s.links["link_id"].iloc[i]) for i in bad]
    s = s.subset(~np.isin(s.link, bad))
    tmp = p.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        x=s.x,
        m=s.m,
        r=s.r,
        link=s.link,
        day=s.day,
        time=s.time,
        split=s.split,
        links=s.links.to_json(),
        dropped=np.array(dropped, dtype=object),
    )
    tmp.replace(p)
    return s, dropped


def run_job(job: Job) -> list[dict]:
    p = _cache() / f"{job.key()}.json"
    if p.exists():
        return json.loads(p.read_text())
    s, _ = samples(job.dataset, job.noise_db)
    mask = budget_mask(s, job.budget, seed=0) if job.budget < 1 else None
    cfg = TrainConfig().quick() if job.quick else TrainConfig()
    t0 = time.perf_counter()
    out = fit_arms(s, job.seed, mask, cfg, job.arms)
    # test predictions, for the analysis by intensity and by event
    te = s.split == 2
    np.savez_compressed(
        _cache() / f"{job.key()}_pred.npz",
        r=s.r[te],
        link=s.link[te],
        time=s.time[te].astype("datetime64[m]").astype(np.int64),
        **{f"pred_{a}": v.astype(np.float32) for a, v in out["pred"].items()},
        **{f"gate_{a}": v.astype(np.float32) for a, v in out["gate"].items()},
    )
    rows = []
    for arm, m in out["metrics"].items():
        rows.append(
            {
                "stage": job.stage,
                "dataset": job.dataset,
                "seed": job.seed,
                "budget": job.budget,
                "noise_db": job.noise_db,
                "arm": arm,
                "n_train": out["n_train"],
                "n_test": out["n_test"],
                "seconds": out["seconds"].get(arm, np.nan),
                **m,
            }
        )
    rows[0]["job_seconds"] = time.perf_counter() - t0
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(p)
    return rows


def jobs(stage: str, quick: bool = False) -> list[Job]:
    data = available() + list(SIMS)
    seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS
    if stage == "main":
        return [Job(d, s, stage="main", quick=quick) for d in data for s in seeds]
    if stage == "budget":
        arms = ("pl_cal", "gru", "hybrid_gate", "hybrid_phased")
        return [
            Job(d, s, b, arms, stage="budget", quick=quick)
            for d in [*available(), "sim_ar1_waa"]
            for b in BUDGETS
            for s in seeds
        ]
    if stage == "noise":
        arms3: tuple[str, ...] = ("pl_cal", "gru", "hybrid_phased")
        return [
            Job(d, REPORT_SEEDS[0], 1.0, arms3, n, stage="noise", quick=quick)
            for d in ("sim_iid", "sim_ar1_waa")
            for n in NOISES
        ]
    raise ValueError(stage)


def run(stages=("main", "budget", "noise"), workers: int = 4, quick: bool = False):
    todo = [j for st in stages for j in jobs(st, quick)]
    # samples are built once, in this process, before the pool starts
    for d in sorted({(j.dataset, j.noise_db) for j in todo}, key=str):
        samples(*d)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PHYSPRIOR_NO_JULIA", "1")
    rows: list[dict] = []
    pending = list(todo)
    for attempt in range(10):
        try:
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                for job, res in zip(pending, ex.map(run_job, pending), strict=True):
                    rows.extend(res)
                    log.info(
                        "done %s %s seed %d budget %g",
                        job.stage,
                        job.dataset,
                        job.seed,
                        job.budget,
                    )
            break
        except BrokenProcessPool:
            log.warning("pool broke (attempt %d); resubmitting", attempt + 1)
            rows = []
            pending = list(todo)
    write_results(rows, quick)
    if not quick and "main" in stages:
        from .analysis import run as analyse

        analyse()
    return rows


def write_results(rows: list[dict], quick: bool = False) -> None:
    out = get_settings().results("cml_quick" if quick else "cml")
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    info, drops = [], []
    for d in sorted(set(df["dataset"])):
        s, dropped = samples(d)
        info.append(
            {
                "dataset": d,
                "n_links": int(s.links.shape[0] - len(dropped)),
                "n_samples": len(s),
                "bin_steps": int(s.m),
                "wet_fraction": float((s.r > 0.1).mean()),
                "mean_rain_mmh": float(s.r.mean()),
                "freq_min": float(s.links["freq_ghz"].min()),
                "freq_max": float(s.links["freq_ghz"].max()),
                "len_min": float(s.links["length_km"].min()),
                "len_max": float(s.links["length_km"].max()),
            }
        )
        drops += [{"dataset": d, "link_id": x} for x in dropped]
    pd.DataFrame(info).to_csv(out / "datasets.csv", index=False)
    pd.DataFrame(drops, columns=["dataset", "link_id"]).to_csv(
        out / "dropped_links.csv", index=False
    )
