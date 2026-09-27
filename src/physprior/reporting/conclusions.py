"""Turn `results/` into a verdict, in code, so a notebook cannot conclude
something the numbers do not say.

Every notebook ends in a conclusion. Writing that conclusion by hand is how a
project ends up claiming a win it no longer has, so the text is rendered from
the same CSVs the figures are drawn from, and the verdict has three possible
values, not two:

    WIN       one arm is better and the gap survives the seed spread
    TIE       the gap is inside the seed spread -- reported as a result,
              not rounded into a win
    (none)    the question was not run on this track

The seed spread is the point. With three reporting seeds a 2x difference on
one metric is frequently nothing at all, and an arm that "wins" six tracks by
less than its own seed-to-seed scatter has not won anything. `decisive` is
what separates the two, and the notebooks print it.

The oracle is excluded from the competition (invariant 8: it is the published
law with published constants, a ceiling rather than a competitor), but it is
still scored, and an arm that beats it out of range is flagged.

**Which column is the answer.** In `sweep_budget` and `sweep_noise`,
`score()` is called with the TRAINING indices as `idx_in`, so `nrmse_in` is
the fit to the data the arm was given and `nrmse_out` is the held-out
remainder. Every question here that asks "which arm generalises" therefore
reads `nrmse_out`; reading `nrmse_in` would reward exact interpolation and
hand `physics` a 4.7e6x "win" on two points for fitting one parameter
through them. Only the extrapolation study, whose split is by RANGE rather
than at random, uses `nrmse_in` for anything -- and there it is the in-range
half, which `report.py` already reports beside the out-of-range half.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json

# The real-data benchmark tracks, by problem.
TRACKS_BY_PROBLEM = {
    "gravity": ["gravity/kepler", "gravity/pulsar_spindown"],
    "relativity": ["relativity/gw150914"],
    "quantum": ["quantum/hydrogen", "quantum/helium", "quantum/cmb"],
    "fields": ["fields/weather/july"],
}
ALL_TRACKS = [t for ts in TRACKS_BY_PROBLEM.values() for t in ts]
COMPETITORS = ("physics", "pinn", "sr", "nn")


@dataclass
class Verdict:
    """One track, one protocol question, one answer."""

    track: str
    question: str
    winner: str
    runner_up: str
    winner_score: float
    runner_score: float
    spread: float  # pooled seed-to-seed std of the two
    decisive: bool  # is the gap bigger than that spread?
    lower_is_better: bool = True
    note: str = ""

    @property
    def margin(self) -> float:
        """How many times better the winner is. 1.0 means no difference."""
        if self.winner_score == 0 or self.runner_score == 0:
            return float("inf")
        ratio = self.runner_score / self.winner_score
        return ratio if self.lower_is_better else 1 / ratio

    def as_row(self) -> dict:
        return {
            "track": self.track,
            "question": self.question,
            "winner": self.winner,
            "runner-up": self.runner_up,
            "margin": f"{self.margin:.3g}x" if np.isfinite(self.margin) else "-",
            "seed spread": f"{self.spread:.2g}",
            "verdict": "WIN" if self.decisive else "TIE (within seed spread)",
        }


def _read(track: str, name: str) -> pd.DataFrame | None:
    path = get_settings().results_dir / track / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _rank(df: pd.DataFrame, metric: str, arms=COMPETITORS) -> Verdict | None:
    """Rank the arms on one metric, lowest first, and say whether the gap
    between the best two is bigger than their pooled seed scatter."""
    g = df[df.arm.isin(arms)].groupby("arm")[metric].agg(["mean", "std"])
    g = g.dropna(subset=["mean"])
    if len(g) < 2:
        return None
    order = g["mean"].sort_values().index.tolist()
    win, run = order[0], order[1]
    spread = float(np.hypot(g.loc[win, "std"] or 0.0, g.loc[run, "std"] or 0.0))
    gap = float(g.loc[run, "mean"] - g.loc[win, "mean"])
    return Verdict(
        track="",
        question="",
        winner=win,
        runner_up=run,
        winner_score=float(g.loc[win, "mean"]),
        runner_score=float(g.loc[run, "mean"]),
        spread=spread,
        decisive=gap > spread,
    )


def _with(v: Verdict | None, track: str, question: str, note: str = "") -> list:
    if v is None:
        return []
    v.track, v.question, v.note = track, question, note
    return [v]


def _parameter_recovery(track: str) -> list:
    """Which arm returns the physical constant closest to the published one.

    Only the arms that return a constant at all can enter, which is the
    result: `nn` cannot compete because it has nothing to report.
    """
    df = _read(track, "sweep_budget")
    if df is None:
        return []
    df = df[df.n_train == df.n_train.max()]
    cols = [c for c in df.columns if c.startswith("err_")]
    out: list = []
    for col in cols:
        sub = df[["arm", col]].copy()
        sub[col] = sub[col].abs()
        sub = sub[sub.arm != "oracle"].dropna()
        if sub.arm.nunique() < 2:
            continue
        v = _rank(sub, col, arms=tuple(sub.arm.unique()))
        quantity = col.removeprefix("err_").removesuffix("_pct")
        out += _with(v, track, f"parameter recovery ({quantity})")
    return out


def _law_recovery(track: str) -> dict:
    """Not a race -- a capability question, answered per arm.

    `sr` is the only arm that can return a law it was not given, so the
    question is whether the law it found IS the published one. That is
    recorded by the track's own check (`law_check`), measured on the
    function rather than parsed from the string.
    """
    try:
        head = load_json(track, "meta").get("headline", {})
    except FileNotFoundError:
        return {}
    out = {}
    for arm, rec in head.items():
        if not isinstance(rec, dict) or "expression" not in rec:
            continue
        out[arm] = {
            "expression": rec.get("expression"),
            "returns a law": rec.get("expression") is not None,
        }
    check = head.get("sr", {}).get("law_check") or {}
    flags = [k for k in ("is_planck_form", "is_bohr_form") if k in check]
    if flags:
        out.setdefault("sr", {})["matches the published law"] = bool(check[flags[0]])
    elif check.get("exponent") is not None:
        published = check.get("kepler_exponent", check.get("newtonian_exponent"))
        if published is not None:
            out.setdefault("sr", {})["matches the published law"] = bool(
                abs(check["exponent"] - published) < 0.01
            )
    elif "sr" in out:
        out["sr"]["matches the published law"] = False
    return out


def verdicts(track: str) -> list:
    """Every protocol question that was run on this track."""
    out: list = []
    budget = _read(track, "sweep_budget")
    if budget is not None:
        big = budget[budget.n_train == budget.n_train.max()]
        out += _with(
            _rank(big, "nrmse_out"),
            track,
            "interpolation",
            note="held-out points inside the training range",
        )
        small = budget[budget.n_train == budget.n_train.min()]
        out += _with(
            _rank(small, "nrmse_out"),
            track,
            "data efficiency",
            note=f"at the smallest budget, n_train={int(budget.n_train.min())}",
        )
    noise = _read(track, "sweep_noise")
    if noise is not None:
        worst = noise[noise.noise_frac == noise.noise_frac.max()]
        out += _with(
            _rank(worst, "nrmse_out"),
            track,
            "noise",
            note=f"at the heaviest corruption, {noise.noise_frac.max():g}",
        )
    extra = _read(track, "extrapolation")
    if extra is not None:
        out += _with(_rank(extra, "nrmse_out"), track, "extrapolation")
    out += _parameter_recovery(track)
    return out


def verdict_frame(scope: str = "all") -> pd.DataFrame:
    """The table a notebook displays: track x question -> winner, margin,
    and whether the margin survives the seed spread."""
    rows = [v.as_row() for t in _scope_tracks(scope) for v in verdicts(t)]
    return pd.DataFrame(rows)


def _scope_tracks(scope: str) -> list:
    if scope == "all":
        return ALL_TRACKS
    if scope in TRACKS_BY_PROBLEM:
        return TRACKS_BY_PROBLEM[scope]
    return [scope]


@dataclass
class Summary:
    """The variables the prose is rendered from. Nothing is typed by hand."""

    scope: str
    n_questions: int = 0
    n_decisive: int = 0
    wins: dict = field(default_factory=dict)  # arm -> decisive wins
    won_what: dict = field(default_factory=dict)  # arm -> [(track, question)]
    ties: int = 0
    biggest: Verdict | None = None
    pinn: dict = field(default_factory=dict)  # arm-specific tally for `pinn`
    beats_oracle: list = field(default_factory=list)


def summarise(scope: str = "all") -> Summary:
    vs = [v for t in _scope_tracks(scope) for v in verdicts(t)]
    s = Summary(scope=scope, n_questions=len(vs))
    for v in vs:
        if v.decisive:
            s.n_decisive += 1
            s.wins[v.winner] = s.wins.get(v.winner, 0) + 1
            s.won_what.setdefault(v.winner, []).append((v.track, v.question))
        else:
            s.ties += 1
    finite = [v for v in vs if v.decisive and np.isfinite(v.margin)]
    s.biggest = max(finite, key=lambda v: v.margin) if finite else None
    s.pinn = {
        "wins": sum(1 for v in vs if v.decisive and v.winner == "pinn"),
        "losses": sum(1 for v in vs if v.decisive and v.winner != "pinn"),
        "ties": sum(1 for v in vs if not v.decisive),
    }
    s.beats_oracle = _oracle_flags(scope)
    return s


# An arm within a few per cent of the oracle has not "beaten" it; that is
# seed scatter. Only a margin a reader would act on is worth flagging.
ORACLE_MARGIN = 1.05


def _oracle_flags(scope: str) -> list:
    """An arm beating the oracle out of range needs an explanation
    (invariant 8). The report records one; this surfaces it."""
    out = []
    for track in _scope_tracks(scope):
        df = _read(track, "extrapolation")
        if df is None:
            continue
        g = df.groupby("arm")["nrmse_out"].mean()
        if "oracle" not in g.index:
            continue
        for arm in COMPETITORS:
            if arm in g.index and g["oracle"] / g[arm] > ORACLE_MARGIN:
                out.append((track, arm, float(g["oracle"] / g[arm])))
    return out


def conclusion_markdown(scope: str = "all") -> str:
    """Three to five sentences, rendered from `summarise`.

    If nothing is decisive, that is what it says. A notebook that cannot
    reach a conclusion reports exactly that, as a result.
    """
    s = summarise(scope)
    if s.n_questions == 0:
        return (
            "**No conclusion.** No results are on disk for "
            f"`{scope}` — run `physprior run {scope}` first."
        )
    if s.n_decisive == 0:
        return (
            f"**No conclusion is available for `{scope}`.** All "
            f"{s.n_questions} protocol questions separate the arms by less "
            "than the seed-to-seed spread on the reporting seeds "
            "(11/23/42), so the differences are not measurable with three "
            "seeds. That is the result: on this evidence the arms are "
            "indistinguishable here, and any ranking would be noise."
        )
    ranked = sorted(s.wins.items(), key=lambda kv: -kv[1])
    top_n = ranked[0][1]
    leaders = [a for a, n in ranked if n == top_n]
    others = ", ".join(f"`{a}` {n}" for a, n in ranked if a not in leaders)
    lines = [
        f"Across {s.n_questions} protocol questions on "
        f"{len(_scope_tracks(scope))} track(s), **{s.n_decisive} produced a "
        f"decisive answer** and {s.ties} were inside the seed-to-seed spread "
        "and are reported as ties rather than rounded into wins.",
    ]
    if len(leaders) == 1:
        tail = f" ({others})" if others else ""
        lines.append(
            f"**`{leaders[0]}` takes {top_n} of the decisive question(s)**{tail}."
        )
    else:
        names = ", ".join(f"`{a}`" for a in leaders)
        lines.append(
            f"**No arm leads**: {names} take {top_n} decisive question(s) "
            "each, so the decisive questions split rather than pointing at a "
            "winner."
        )
    if s.biggest is not None:
        b = s.biggest
        lines.append(
            f"The largest single gap is on **{b.track} / {b.question}**, where "
            f"`{b.winner}` beats `{b.runner_up}` by **{b.margin:.3g}x**."
        )
    p = s.pinn
    if p["wins"] or p["losses"]:
        lines.append(
            f"The `pinn` arm — the one this project is built around — wins "
            f"**{p['wins']}**, loses **{p['losses']}** and ties **{p['ties']}** "
            "of these, which is the number to read before any claim about "
            "what a physics prior buys."
        )
    if s.beats_oracle:
        wh = "; ".join(f"`{a}` on {t} by {r:.3g}x" for t, a, r in s.beats_oracle)
        lines.append(
            f"An arm beats the **oracle** out of range ({wh}): the oracle "
            "carries the *published* constant, so this is a claim about that "
            "constant, and the track's `meta.json` records the explanation."
        )
    return " ".join(lines)


def _falsifier(arm: str, n: int, where: str) -> str:
    """What observation would take these wins away from this arm.

    Different arms lose for different reasons, so a single sentence reused
    for all of them says nothing.
    """
    if arm == "nn":
        return (
            f"`nn` wins {n} question(s) ({where}): the black box is only as "
            "strong as the grid it was tuned on, so a wider architecture "
            "search for the other arms — or simply more data, which is what "
            "it is exploiting — would move this."
        )
    if arm == "sr":
        return (
            f"`sr` wins {n} question(s) ({where}): symbolic regression's "
            "reach is set by its operator set, which is a prior. Adding or "
            "removing an operator changes what it can find at all, so these "
            "wins are contingent on a choice made in the track definition."
        )
    if arm == "physics":
        return (
            f"`physics` wins {n} question(s) ({where}): it assumes the "
            "published law is right. A track where the law is incomplete — "
            "a truncated expansion, a neglected term — removes that "
            "advantage, which is exactly what the `pinn` arm exists to test."
        )
    if arm == "pinn":
        return (
            f"`pinn` wins {n} question(s) ({where}): the neural correction "
            "pays only when the law has something left over for it to "
            "absorb. Fitting the same track with the complete law would "
            "remove the residual it is exploiting and the win with it."
        )
    return (
        f"`{arm}` wins {n} question(s) ({where}): re-tuning the rival arms "
        "on the TUNING seeds (3/7/19) could close the gap."
    )


def what_would_change_this(scope: str = "all") -> list:
    """One line per claim: what observation would overturn it.

    A conclusion that nothing could falsify is not a result, so each claim
    above is paired here with the measurement that would undo it.
    """
    s = summarise(scope)
    out = []
    if s.n_questions == 0:
        return ["Running the track at all would change this."]
    if s.n_decisive == 0:
        return [
            "More reporting seeds: with three seeds only gaps larger than the "
            "seed scatter are visible, and a real effect smaller than that "
            "would still be hiding here.",
            "A wider sweep: the arms may separate at a budget, noise level or "
            "extrapolation distance outside the range swept.",
        ]
    for arm, n in sorted(s.wins.items(), key=lambda kv: -kv[1]):
        where = "; ".join(f"{t.split('/')[-1]} {q}" for t, q in s.won_what[arm])
        out.append(_falsifier(arm, n, where))
    if s.ties:
        out.append(
            f"{s.ties} question(s) are ties: more seeds, or a lower-variance "
            "fit (ensembling, early stopping), would decide them either way."
        )
    if s.biggest is not None:
        b = s.biggest
        out.append(
            f"The {b.margin:.3g}x gap on {b.track} / {b.question} rests on "
            f"three seeds with a spread of {b.spread:.2g}; it would be "
            "overturned by a seed set on which `"
            f"{b.runner_up}` closes to within that spread, which is why the "
            "margin and the spread are printed side by side above."
        )
    if s.beats_oracle:
        out.append(
            "The oracle being beaten would be overturned by a revised "
            "published constant: the oracle is only as good as the literature "
            "value it carries."
        )
    return out
