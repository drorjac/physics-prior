"""The two trials end to end: tune the baselines, search, select, report.

Seeds follow the repository rule. Baselines are tuned and the search is
scored on the meta-training tasks with tuning seeds only; one rule is
selected on the meta-training tasks with all three tuning seeds; every
reported number is on held-out tasks (and, separately, the meta-training
tasks) with the reporting seeds 11, 23, 42.

Outputs, per trial, in results/lossdisc/<trial>/:

    baselines.csv   the tuning grid of every baseline family
    search.csv      every rule the search trained, with its fitness
    history.csv     the best rule after each generation
    selection.csv   the top of the search re-scored on seeds 3, 7, 19
    report.csv      one row per (method, task, seed) on the reporting seeds
    summary.json    the chosen rules and the paired comparison
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from physprior.config import get_settings
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS
from physprior.symbolic.expressions import Node, const

from . import rules as R
from .pinn import TrainConfig
from .search import Runner, SearchConfig, evaluate_rules, search
from .tasks import FORWARD_TEST, FORWARD_TRAIN, INVERSE_TEST, INVERSE_TRAIN

log = logging.getLogger(__name__)

SEARCH_SEEDS = TUNE_SEEDS[:2]  # 3, 7; seed 19 is kept back for selection
N_SELECT = 6
FAIL = {"forward": 0.3, "inverse": 0.1}  # error above which a run has failed


@dataclass(frozen=True)
class Trial:
    name: str
    train: tuple
    test: tuple
    n_features: int
    names: tuple[str, ...]
    grids: dict  # family -> list of (label, tree)
    normalise: bool  # forward: learned rules are mean-normalised


def trials() -> dict[str, Trial]:
    return {
        "forward": Trial(
            "forward",
            FORWARD_TRAIN,
            FORWARD_TEST,
            4,
            R.RESIDUAL_FEATURES,
            {
                "uniform": [("g = 0", R.uniform().tree)],
                "causal": [
                    (f"eps = {e:g}", R.causal(e).tree) for e in (0.3, 1, 3, 10, 30, 100)
                ],
            },
            True,
        ),
        "inverse": Trial(
            "inverse",
            INVERSE_TRAIN,
            INVERSE_TEST,
            3,
            R.BALANCE_FEATURES,
            {
                "fixed": [
                    (f"lambda = {v:g}", R.fixed(v).tree)
                    for v in (0.01, 0.03, 0.1, 0.3, 1, 3, 10)
                ],
                "gradnorm": [
                    (f"scale = {v:g}", R.gradnorm(v).tree) for v in (0.1, 1, 10)
                ],
            },
            False,
        ),
    }


def _out_dir(trial: str, quick: bool):
    sub = f"lossdisc/{trial}" + ("_quick" if quick else "")
    return get_settings().results(sub)


def _normalise_for(trial: Trial, family: str) -> bool:
    """The published baselines run as published (causal weights are not
    normalised); learned rules run with the trial's normalisation."""
    return trial.normalise if family == "learned" else False


def run_trial(
    trial: Trial,
    *,
    quick: bool = False,
    workers: int = 7,
    search_cfg: SearchConfig | None = None,
) -> dict:
    cfg = TrainConfig().quick() if quick else TrainConfig()
    scfg = search_cfg or (
        SearchConfig(population=6, generations=2, elite=2) if quick else SearchConfig()
    )
    train = trial.train[:1] if quick else trial.train
    test = trial.test[:2] if quick else trial.test
    report_seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS
    out = _out_dir(trial.name, quick)
    t_name = trial.name

    with Runner(workers) as runner:
        # 1. baselines, tuned on the meta-training tasks and tuning seeds
        rows, chosen = [], {}
        for fam, grid in trial.grids.items():
            scored = evaluate_rules(
                runner,
                t_name,
                [t for _, t in grid],
                train,
                TUNE_SEEDS,
                cfg,
                _normalise_for(trial, fam),
            )
            for (lab, _tree), (fit, _) in zip(grid, scored, strict=True):
                rows.append({"family": fam, "setting": lab, "fitness": fit})
            i = int(np.argmin([f for f, _ in scored]))
            chosen[fam] = (grid[i][0], grid[i][1], scored[i][0])
            log.info("%s: %s tuned to %s (%.3f)", t_name, fam, grid[i][0], scored[i][0])
        pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False)

        # 2. the search, on seeds 3 and 7; the tuned baselines are in the
        #    initial population, so the search starts where the literature is
        def fitness(trees: list[Node]) -> list[float]:
            res = evaluate_rules(
                runner, t_name, trees, train, SEARCH_SEEDS, cfg, trial.normalise
            )
            return [f for f, _ in res]

        seeds_init = [const(0.0)] + [tree for _, tree, _ in chosen.values()]
        result = search(
            fitness,
            trial.n_features,
            seeds_init=seeds_init,
            config=scfg,
            names=trial.names,
            log=lambda m: log.info("%s %s", t_name, m),
        )
        pd.DataFrame(
            [
                {"rule": e.label, "size": e.size, "fitness": e.fitness}
                for e in sorted(result.evaluated, key=lambda e: e.fitness)
            ]
        ).to_csv(out / "search.csv", index=False)
        pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)

        # 3. selection: the top of the search re-scored with seed 19 added
        top = result.best(N_SELECT, scfg.parsimony)
        rescored = evaluate_rules(
            runner,
            t_name,
            [e.tree for e in top],
            train,
            TUNE_SEEDS,
            cfg,
            trial.normalise,
        )
        sel = pd.DataFrame(
            [
                {
                    "rule": e.label,
                    "size": e.size,
                    "fitness_search": e.fitness,
                    "fitness_select": f,
                    "cost": f + scfg.parsimony * e.size,
                }
                for e, (f, _) in zip(top, rescored, strict=True)
            ]
        )
        sel.to_csv(out / "selection.csv", index=False)
        j = int(np.argmin(sel["cost"].to_numpy()))
        learned = top[j].tree
        log.info("%s: selected %s", t_name, top[j].label)

        # 4. report: held-out tasks and the meta-training tasks, report seeds
        methods = {fam: (lab, tree) for fam, (lab, tree, _) in chosen.items()}
        methods["learned"] = (top[j].label, learned)
        rep_rows = []
        for split, tasks in (("test", test), ("train", train)):
            for fam, (lab, tree) in methods.items():
                _, res = evaluate_rules(
                    runner,
                    t_name,
                    [tree],
                    tasks,
                    report_seeds,
                    cfg,
                    _normalise_for(trial, fam),
                )[0]
                for r in res:
                    rep_rows.append(
                        {"split": split, "method": fam, "setting": lab, **r}
                    )
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(out / "report.csv", index=False)

    summary = {
        "trial": t_name,
        "quick": quick,
        "train_config": asdict(cfg),
        "search_config": asdict(scfg),
        "train_tasks": [t.name for t in train],
        "test_tasks": [t.name for t in test],
        "tuned": {
            fam: {"setting": lab, "fitness": f} for fam, (lab, _, f) in chosen.items()
        },
        "learned": {
            "rule": top[j].label,
            "size": top[j].size,
            "tree": R.to_json(learned),
        },
        "n_rules_trained": len(result.evaluated),
        "search_seconds": result.seconds,
        "comparison": compare(rep, t_name),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _metric(trial: str, df: pd.DataFrame) -> pd.Series:
    if trial == "forward":
        return np.log10(df["err"].clip(lower=1e-12))
    return 0.5 * (
        np.log10(df["err_k"].clip(lower=1e-12))
        + np.log10(df["err_u"].clip(lower=1e-12))
    )


def compare(rep: pd.DataFrame, trial: str) -> dict:
    """Per split and method: medians, failure rate, and the paired test of the
    learned rule against each baseline over (task, seed) pairs."""
    err = "err" if trial == "forward" else "err_k"
    out: dict = {}
    for split, d in rep.groupby("split"):
        d = d.assign(score=_metric(trial, d))
        stats = {}
        for m, g in d.groupby("method"):
            stats[m] = {
                "median_err": float(g[err].median()),
                "median_err_u": float(g["err_u"].median()) if "err_u" in g else None,
                "mean_log10": float(g["score"].mean()),
                "fail_rate": float((g[err] > FAIL[trial]).mean()),
                "n": len(g),
            }
        piv = d.pivot_table(index=["task", "seed"], columns="method", values="score")
        paired = {}
        for m in piv.columns:
            if m == "learned":
                continue
            diff = (piv["learned"] - piv[m]).dropna().to_numpy()
            p = (
                float(wilcoxon(diff).pvalue)
                if len(diff) >= 6 and np.any(diff)
                else math.nan
            )
            paired[m] = {
                "wins": int((diff < 0).sum()),
                "losses": int((diff > 0).sum()),
                "mean_gain_decades": float(-diff.mean()),
                "wilcoxon_p": p,
            }
        out[split] = {"methods": stats, "learned_vs": paired}
    return out


def run(quick: bool = False, workers: int = 7, only: str | None = None) -> dict:
    out = {}
    for name, trial in trials().items():
        if only and name != only:
            continue
        out[name] = run_trial(trial, quick=quick, workers=workers)
    return out
