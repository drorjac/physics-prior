"""docs/lossdisc/README.md and figures/lossdisc/, generated from results/lossdisc.

Every number on the page is read from a results file. Regenerate with
`physprior lossdisc --doc-only`.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.viz.palette import ARM_PALETTE
from physprior.viz.plots import GRID, INK_MUTED, SURFACE, _style, use_style

from .rules import BalanceRule, ResidualRule, from_json
from .study import FAIL, trials

DOC = "lossdisc/README.md"
TRACK = "lossdisc"
COLOR = {
    "uniform": INK_MUTED,
    "causal": ARM_PALETTE[0],
    "fixed": ARM_PALETTE[0],
    "gradnorm": ARM_PALETTE[1],
    "learned": ARM_PALETTE[2],
}
MARKER = {"uniform": "s", "causal": "D", "fixed": "D", "gradnorm": "^", "learned": "o"}
LABEL = {
    "uniform": "plain PINN",
    "causal": "causal weights (tuned eps)",
    "fixed": "fixed lambda (tuned)",
    "gradnorm": "gradient-norm balance",
    "learned": "SR-learned rule",
}
ERR = {"forward": "err", "inverse": "err_k"}
ERR_NAME = {"forward": "relative L2 error in u", "inverse": "relative error in k"}


def _dir(trial: str):
    return get_settings().results_dir / TRACK / trial


def load(trial: str) -> tuple[dict, pd.DataFrame]:
    d = _dir(trial)
    return json.loads((d / "summary.json").read_text()), pd.read_csv(d / "report.csv")


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.{nd}g}"


def _p(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.2g}" if x >= 1e-3 else f"{x:.1e}"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save(fig, name: str):
    p = get_settings().figures(TRACK) / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_tasks(trial: str):
    """Error per held-out task: every seed as a small mark, the median large."""
    use_style()
    _, rep = load(trial)
    d = rep[rep["split"] == "test"]
    tasks = list(dict.fromkeys(d["task"]))
    methods = list(dict.fromkeys(d["method"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(tasks) + 1.4))
    off = np.linspace(-0.22, 0.22, len(methods))
    for k, m in enumerate(methods):
        g = d[d["method"] == m]
        for i, t in enumerate(tasks):
            v = g[g["task"] == t][ERR[trial]].to_numpy()
            y = len(tasks) - 1 - i + off[k]
            ax.scatter(v, np.full(len(v), y), s=14, color=COLOR[m], alpha=0.45, lw=0)
            ax.scatter(
                [np.median(v)],
                [y],
                s=58,
                color=COLOR[m],
                marker=MARKER[m],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
                label=LABEL[m] if i == 0 else None,
            )
    ax.axvline(FAIL[trial], color=GRID, lw=1.5, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(tasks)), tasks[::-1])
    _style(ax, title=None, xlabel=ERR_NAME[trial] + " (log scale)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    return _save(fig, f"{trial}_tasks")


def fig_search(trial: str):
    """Best fitness per generation against the tuned baselines."""
    use_style()
    summ, _ = load(trial)
    h = pd.read_csv(_dir(trial) / "history.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(
        h["generation"],
        h["best_fitness"],
        color=COLOR["learned"],
        marker="o",
        lw=2,
        label="best rule so far (seeds 3, 7)",
    )
    for fam, v in summ["tuned"].items():
        ax.axhline(
            v["fitness"],
            color=COLOR[fam],
            lw=1.5,
            ls="--",
            label=f"{LABEL[fam]} (seeds 3, 7, 19)",
        )
    _style(ax, xlabel="generation", ylabel="mean log10 error, meta-training tasks")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, f"{trial}_search")


def fig_forward_weights(rule_tree):
    """The learned residual weights over time at three points of training, for
    a residual uniform in t and for one that grows towards the end of the
    window (a solution that is right early and wrong late)."""
    use_style()
    rule = ResidualRule(rule_tree)
    x = (np.arange(128) + 0.5) / 128
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    profiles = (("residual uniform in t", np.ones_like(x)), ("residual ~ t^4", x**4))
    for ax, (title, r2) in zip(axes, profiles, strict=True):
        for tau, ls in ((0.0, ":"), (0.5, "--"), (0.95, "-")):
            try:
                w = rule.weights(x, r2, tau)
            except ValueError:
                continue
            ax.plot(x, w, color=COLOR["learned"], ls=ls, lw=2, label=f"tau = {tau:g}")
        _style(ax, title=title, xlabel="t / T")
        ax.set_yscale("log")
    axes[0].set_ylabel("weight w(t)")
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, "forward_weights")


def fig_inverse_schedule(rule_tree):
    """The learned lambda as a function of training progress and of rho."""
    use_style()
    rule = BalanceRule(rule_tree)
    tau = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for rho, ls in ((-2.0, ":"), (0.0, "--"), (2.0, "-")):
        lam = []
        for t in tau:
            try:
                lam.append(rule.weight(t, rho, 0.0))
            except ValueError:
                lam.append(np.nan)
        ax.plot(tau, lam, color=COLOR["learned"], ls=ls, lw=2, label=f"rho = {rho:g}")
    ax.set_yscale("log")
    _style(ax, xlabel="training progress tau", ylabel="lambda")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "inverse_schedule")


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def _table(trial: str, summ: dict, split: str) -> list[str]:
    c = summ["comparison"][split]
    unit = "median err" if trial == "forward" else "median err in k"
    inverse = trial == "inverse"
    lines = [
        f"| method | setting | {unit} | "
        + ("median err in u | " if inverse else "")
        + "mean log10 error | failed | runs |",
        "|---|---|---|---|---|" + ("---|---|" if inverse else "---|"),
    ]
    settings = {fam: v["setting"] for fam, v in summ["tuned"].items()}
    settings["learned"] = f"`{summ['learned']['rule']}`"
    for m in sorted(c["methods"], key=list(LABEL).index):
        s = c["methods"][m]
        eu = f"{_fmt(s['median_err_u'])} | " if inverse else ""
        lines.append(
            f"| {LABEL[m]} | {settings.get(m, '')} | {_fmt(s['median_err'])} | "
            f"{eu}{_fmt(s['mean_log10'])} | {s['fail_rate']:.0%} | {s['n']} |"
        )
    return lines


def _paired(summ: dict, split: str) -> list[str]:
    lines = [
        "| learned rule against | wins | losses | mean gain (decades) | Wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for m, v in summ["comparison"][split]["learned_vs"].items():
        lines.append(
            f"| {LABEL[m]} | {v['wins']} | {v['losses']} | "
            f"{_fmt(v['mean_gain_decades'], 2)} | {_p(v['wilcoxon_p'])} |"
        )
    return lines


def _constant_note(trial: str, summ: dict) -> list[str]:
    """Say so when the search returned a constant: then it found no schedule
    or adaptive rule better than a fixed setting."""
    tree = summ["learned"]["tree"]
    if tree[0] != "c":
        return []
    if trial == "inverse":
        what = f"a fixed weight, lambda = {np.exp(tree[1]):.3g}"
    else:
        what = "uniform weights"
    return [
        f"The selected rule is a constant, that is {what}. The search found "
        "no rule that depends on the training state and does better on "
        "these tasks.",
        "",
    ]


def _section(trial: str, title: str, intro: str) -> list[str]:
    summ, _ = load(trial)
    L = [f"## {title}", "", intro, ""]
    L += [
        f"Meta-training tasks: {', '.join(summ['train_tasks'])}. "
        f"Held-out tasks: {', '.join(summ['test_tasks'])}. "
        f"The search trained {summ['n_rules_trained']} distinct rules in "
        f"{summ['search_seconds'] / 60:.0f} minutes. Selected rule "
        f"({summ['learned']['size']} node{'s' if summ['learned']['size'] > 1 else ''}):",
        "",
        f"    {summ['learned']['rule']}",
        "",
        *_constant_note(trial, summ),
        "Held-out tasks, reporting seeds 11, 23, 42:",
        "",
        *_table(trial, summ, "test"),
        "",
        *_paired(summ, "test"),
        "",
        "Meta-training tasks, reporting seeds (new initialisations and, for the "
        "inverse trial, new noise):",
        "",
        *_table(trial, summ, "train"),
        "",
        *_paired(summ, "train"),
        "",
        f"![{trial}_tasks](../../figures/lossdisc/{trial}_tasks.png)",
        "",
        f"![{trial}_search](../../figures/lossdisc/{trial}_search.png)",
        "",
    ]
    return L


def _first_search() -> list[str]:
    """The first, smaller search (results/lossdisc/v1), kept as run."""
    v1 = get_settings().results_dir / TRACK / "v1"
    rows = []
    for name in ("forward", "inverse"):
        p = v1 / name / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        c = s["comparison"]["test"]
        for m, v in c["learned_vs"].items():
            rows.append(
                f"| {name} | `{s['learned']['rule']}` | {s['n_rules_trained']} | "
                f"{LABEL[m]} | {v['wins']} / {v['losses']} | "
                f"{_fmt(v['mean_gain_decades'], 2)} | {_p(v['wilcoxon_p'])} |"
            )
    if not rows:
        return []
    cfg = json.loads((v1 / "forward" / "summary.json").read_text())["search_config"]
    return [
        "## The first search",
        "",
        f"A first search with population {cfg['population']} and "
        f"{cfg['generations']} generations spent most of its budget on "
        "duplicate children. In trial B it returned a constant, as the "
        "second search did. "
        "The search above redraws duplicates and adds a scaling mutation "
        "(c * subtree). Tasks, seeds and baselines were not changed. The "
        "first search, on the held-out tasks and reporting seeds:",
        "",
        "| trial | rule | rules trained | against | wins / losses | "
        "mean gain (decades) | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def render_doc() -> str:
    t = trials()
    have = {name: (_dir(name) / "summary.json").exists() for name in t}
    for name in t:
        if not have[name]:
            continue
        fig_tasks(name)
        fig_search(name)
    L = [
        "# Loss weightings for PINNs found by symbolic regression",
        "",
        "<!-- Generated by physprior.lossdisc.doc.render_doc() from "
        "results/lossdisc. Do not edit by hand. -->",
        "",
        "A PINN's loss is a weighted sum, and the weighting decides whether "
        "training converges. Here the weighting rule is itself the object of "
        "a symbolic search: a genetic program over expression trees whose "
        "fitness is the error of PINNs trained with the rule. The design and "
        "the protocol are in `docs/plans/LOSSDISC_PLAN.md`; the code is "
        "`src/physprior/lossdisc/`; `physprior lossdisc` reruns it.",
        "",
        "Every method shares the network (tanh MLP, width 32, depth 3), Adam "
        "with a cosine schedule from 1e-2, 3000 steps, and 128 stratified "
        "collocation points redrawn every step. Only the weighting differs. "
        "Baselines are tuned on the same tasks and tuning seeds the search "
        "uses. A run counts as failed above a relative error of "
        f"{FAIL['forward']} (forward) or {FAIL['inverse']} in k (inverse).",
        "",
    ]
    if have["forward"]:
        L += _section(
            "forward",
            "Trial A: residual weights for forward problems",
            "Second-order ODEs over several periods, initial condition built "
            "into the network, no data. A plain PINN fits the early part of "
            "the window and then propagates a wrong solution. The rule maps "
            "(t, C, q, tau) to a per-point weight; `C` is the cumulative "
            "residual before t, `q` the same as a fraction of the total, "
            "`tau` the training progress. Causal weighting is g = -eps C. "
            "Weights are normalised to mean 1, so an additive constant in g "
            "has no effect.",
        )
        fig_forward_weights(from_json(load("forward")[0]["learned"]["tree"]))
        L += ["![forward_weights](../../figures/lossdisc/forward_weights.png)", ""]
    if have["inverse"]:
        L += _section(
            "inverse",
            "Trial B: the data/physics balance for inverse problems",
            "Thirty noisy observations (5 % noise), the form of the ODE known, "
            "k unknown and started at 0.3 of its value. The rule maps "
            "(tau, rho, gamma) to the physics weight lambda; `rho` is "
            "log10(L_data / L_phys) and `gamma` the log10 ratio of the two "
            "gradient norms.",
        )
        tree = load("inverse")[0]["learned"]["tree"]
        if tree[0] != "c":  # a constant has no schedule to draw
            fig_inverse_schedule(from_json(tree))
            L += [
                "![inverse_schedule](../../figures/lossdisc/inverse_schedule.png)",
                "",
            ]
    L += _first_search()
    text = "\n".join(L).rstrip() + "\n"
    p = get_settings().root / "docs" / DOC
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return text
