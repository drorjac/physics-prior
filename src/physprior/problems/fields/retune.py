"""The `pinn` arm's physics weight on the fields tracks, and the re-run.

The same rule as `physprior.benchmark.pinn_tuning`: a weight per case or
scene, chosen on the tuning seeds on a validation block cut from the top of
that track's own extrapolation training range (the highest of the low
stations; the far side of the transmitter's room), then the `pinn` rows of
every committed table recomputed with it. Every other arm's rows are kept.
"""

from __future__ import annotations

import pandas as pd

from physprior.benchmark import pinn_tuning as T
from physprior.benchmark.ablation import tune_dir
from physprior.io import load_json, load_table, save_json, save_table

ARMS = ("pinn",)


def _replace(track: str, name: str, new: pd.DataFrame) -> None:
    old = load_table(track, name)
    save_table(T._replace_arm_rows(old, new, "pinn"), track, name)


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------


def weather(case: str) -> float:
    from physprior.problems.fields import weather as W

    prob, _ = W.problem(case)
    tr = W.track(case)
    chosen, df = T.select_w_phys(prob, W.elevation_split(prob)[0])
    df.to_csv(tune_dir(tr) / "w_phys_selection.csv", index=False)
    prob.pinn_w_phys = chosen
    print(f"[{tr}] w_phys = {chosen:g}", flush=True)

    _replace(tr, "extrapolation", W.study_elevation(prob, arms=ARMS))
    _replace(tr, "blocks", W.study_blocks(prob, arms=ARMS))
    if case == W.SWEEP_CASE:
        from physprior.benchmark.protocol import sweep_budget, sweep_noise

        b = load_table(tr, "sweep_budget")
        _replace(
            tr,
            "sweep_budget",
            sweep_budget(prob, sorted(b.n_train.unique()), arms=ARMS, progress=False),
        )
        nz = load_table(tr, "sweep_noise")
        _replace(
            tr,
            "sweep_noise",
            sweep_noise(
                prob,
                sorted(nz.noise_frac.unique()),
                arms=ARMS,
                n_train=int(nz.n_train.iloc[0]),
                progress=False,
            ),
        )
    meta = load_json(tr, "meta")
    meta["headline"] = W.headline_fits(prob)
    meta["pinn_w_phys"] = chosen
    save_json(meta, tr, "meta")
    return chosen


# --------------------------------------------------------------------------
# radio
# --------------------------------------------------------------------------


def rf() -> dict[str, float]:
    from physprior.benchmark.protocol import fit_arm, split_random
    from physprior.methods.base import REPORT_SEEDS
    from physprior.problems.fields import rf as R

    ppw = int(load_json(R.TRACK, "solver")["ppw"])
    cfgs = load_json(R.TRACK, "nn_cfg")
    budget = load_table(R.TRACK, "sweep_budget")
    noise = load_table(R.TRACK, "sweep_noise")
    head = load_json(R.TRACK, "headline")
    chosen: dict[str, float] = {}
    b_new, n_new, e_new, scs = [], [], [], {}
    for n in ("free", "walls"):
        sc = R.build_scenario(n, ppw)
        scs[n] = sc
        prob = R.make_problem(sc, R._clean(cfgs[n]) | {"epochs": R.EPOCHS})
        prob.pinn_epochs = R.EPOCHS
        w, df = T.select_w_phys(prob, R.extrapolation_split(prob)[0])
        df.to_csv(tune_dir(prob.track) / "w_phys_selection.csv", index=False)
        prob.pinn_w_phys = chosen[n] = w
        print(f"[{prob.track}] w_phys = {w:g}", flush=True)

        sub_b = budget[budget.track == prob.track]
        sub_n = noise[noise.track == prob.track]
        b_new.append(R.sweep_budget(prob, sc, sorted(sub_b.n_train.unique()), ARMS))
        n_new.append(
            R.sweep_noise(
                prob,
                sc,
                sorted(sub_n.noise_db.unique()),
                int(sub_n.n_train.iloc[0]),
                ARMS,
            )
        )
        e_new.append(R.study_extrapolation(prob, sc, ARMS))

        # the headline reconstruction the figures draw, all arms, as sweeps_study
        itr, _ = split_random(len(prob), 128, REPORT_SEEDS[0])
        fits = {
            a: fit_arm(a, prob, itr, REPORT_SEEDS[0]) for a in R.ARMS if a != "oracle"
        }
        R.fig_reconstruction(sc, fits, itr, n)
        head[n] = {
            a: {"params": f.params, "sigma": f.param_sigma, **R.map_scores(f, sc)}
            for a, f in fits.items()
        }
    _replace(R.TRACK, "sweep_budget", pd.concat(b_new, ignore_index=True))
    _replace(R.TRACK, "sweep_noise", pd.concat(n_new, ignore_index=True))
    _replace(R.TRACK, "extrapolation", pd.concat(e_new, ignore_index=True))
    head["pinn_w_phys"] = chosen
    save_json(head, R.TRACK, "headline")

    budget = load_table(R.TRACK, "sweep_budget")
    R.fig_tx(budget, {n: sc.scene for n, sc in scs.items()})
    R.fig_budget(budget)
    R.fig_extrapolation(load_table(R.TRACK, "extrapolation"))
    return chosen
