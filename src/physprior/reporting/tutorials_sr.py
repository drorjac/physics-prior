"""T11 -- how symbolic regression works, built from the in-repo implementation.

    from physprior.reporting.tutorials_sr import t11_how_symbolic_regression_works

The notebook calls `physprior.symbolic` (trees, exhaustive search, the GP,
SINDy) and reads the committed results in results/symbolic/ for the parts
that take minutes to compute.
"""

from __future__ import annotations

from physprior.reporting.cells import SETUP, _nb, code, md


def t11_how_symbolic_regression_works():
    return _nb(
        [
            md("""
# T11 · How symbolic regression works

Symbolic regression (SR) returns a formula rather than a fitted curve. It is
the only arm in this project that can report a law it was not given, and the
only one whose failure mode is "the law was not in its vocabulary".

This notebook builds SR from its parts, with the small implementation in
`physprior.symbolic`:

1. a formula is a **tree**; its size is its **complexity**;
2. the number of trees grows geometrically with size, so **exhaustive search**
   stops being an option after a handful of nodes;
3. **genetic programming** searches the space instead, keeping the best
   formula at every complexity -- the **Pareto front** -- and picks one with
   PySR's **score**;
4. the **operator set is a prior**: Planck's law is out of reach without `exp`;
5. **SINDy** is the sparse-regression alternative for dynamics, and its weak
   point is the derivative.

The theory, with the measured results, is in `docs/theory/symbolic_regression.md`.
"""),
            code(SETUP),
            code("""
import pandas as pd
from physprior.symbolic.expressions import (
    OperatorSet, bi, un, var, const, evaluate, complexity, to_sympy,
    fit_constants, select, scores, noise_floor)
from physprior.symbolic.enumerate import count_by_size, exhaustive_search
from physprior.symbolic.gp import GPConfig, gp_search
from physprior.symbolic.studies import LAWS, make_data, form_check
from physprior.symbolic import sindy
from physprior.io import load_json, load_table
OPS = OperatorSet()   # + - * /  sqrt square cube exp log: the repo's PySR default
print(OPS.label)
"""),
            md("""
## 1 · A formula is a tree

Kepler's third law in years and AU is $P = a^{3/2}$. With the operators above
it can be written as `sqrt(cube(a))`: three nodes. The same function also has
longer spellings (`a * sqrt(a)`, four nodes), and a search will meet all of
them. Complexity here is the node count, which is PySR's default.
"""),
            code("""
kepler = un("sqrt", un("cube", var(0)))
alt = bi("*", var(0), un("sqrt", var(0)))
a = np.geomspace(0.39, 5.2, 6)[:, None]
print(to_sympy(kepler, ["a"]), " complexity", complexity(kepler))
print(to_sympy(alt, ["a"]), "   complexity", complexity(alt))
print(evaluate(kepler, a) - evaluate(alt, a))
"""),
            md("""
Constants are leaves too. A tree fixes the **form**; its constants are then
fitted by ordinary least squares. For Bohr's formula
$\\nu = R_H(1 - 1/n^2)$ the form `c1 - c2/square(n)` has two constants and
six nodes, and Levenberg-Marquardt finds both from a start of 1.
"""),
            code("""
x, y, w, _ = make_data(LAWS["bohr"], 12, 0.0, 3)      # tuning seed, noise-free
bohr_form = bi("-", const(), bi("/", const(), un("square", var(0))))
fitted, loss = fit_constants(bohr_form, x[:, None], y, w)
print(to_sympy(fitted, ["n"]), " loss", loss)
"""),
            md("""
## 2 · The search space

With $L$ leaf types, $U$ unary and $B$ binary operators the number of trees
of size $s$ is $T(1)=L$, $T(s) = U\\,T(s-1) + B\\sum_{i+j=s-1}T(i)T(j)$.
With two leaves (the variable and a constant) and the nine operators above:
"""),
            code("""
counts = count_by_size(15, 2, len(OPS.unary), len(OPS.binary))
print(pd.DataFrame({"size": range(1, 16), "trees": counts}).to_string(index=False))
"""),
            md("""
Every tree with a constant needs its own least-squares fit. Exhaustive search
walks the sizes in order and stops at the first size whose best fit reaches
the measurement-noise floor. For the inverse-square law (size 4) that is
instant; for Bohr (size 6) it is ten thousand fits.
"""),
            code("""
for name in ("inverse_square", "bohr"):
    x, y, w, _ = make_data(LAWS[name], 24, 0.0, 3)
    r = exhaustive_search(x[:, None], y, w, ops=OPS, max_size=7,
                          stop_loss=noise_floor(0.0, len(x)))
    print(f"{name:15s} {r.n_evaluated:6d} trees  {r.seconds:5.1f} s  ->",
          select(r.front).expression())
    print(pd.DataFrame(r.history)[["size", "n_trees", "seconds_cum", "best_loss"]]
          .to_string(index=False))
"""),
            md("""
The committed study repeats this for every law and for a growing vocabulary
(`results/symbolic/time_vs_vocabulary.csv`): each operator added multiplies
the work.
"""),
            code("""
print(load_table("symbolic", "time_vs_vocabulary")[
    ["n_ops", "n_evaluated", "seconds", "expression"]].to_string(index=False))
"""),
            md("""
## 3 · Genetic programming and the Pareto front

GP keeps a population of trees, picks parents by **tournament** (the best of
a few random members, on loss plus a small parsimony penalty), and makes
children by **subtree crossover** and **mutation**. New forms get their
constants fitted. A **hall of fame** holds the best tree at every complexity:
that is the Pareto front, and it is fed back into each generation.

Watch it form on noisy Bohr data (1 % noise, 16 levels, a tuning seed):
"""),
            code("""
x, y, w, _ = make_data(LAWS["bohr"], 16, 0.01, 7)
parsimony = load_json("symbolic", "tune_gp")["chosen_parsimony"]   # tuned on seeds 3/7/19
res = gp_search(x[:, None], y, w, ops=OPS, seed=7,
                config=GPConfig(generations=30, parsimony=parsimony))
fig, ax = plt.subplots(figsize=(6.5, 4))
gens = [0, 2, 5, 10, len(res.fronts) - 1]
for g, c in zip(gens, ["#b7d3f2", "#86b4e8", "#2a78d6", "#153d70", "#0b2240"]):
    fr = res.fronts[g]
    ax.step([f[0] for f in fr], [f[1] for f in fr], where="post", color=c,
            label=f"generation {g}")
ax.set_yscale("log"); ax.set_xlabel("complexity"); ax.set_ylabel("loss"); ax.legend()
plt.show()
"""),
            md("""
### Choosing one expression

PySR scores each step along the front by
$\\text{score}_i = -\\Delta\\log(\\text{loss})/\\Delta\\text{complexity}$: how
much the loss falls per node added. `model_selection="best"` takes the
highest score among expressions within 1.5x of the lowest loss. A law
shows up as a large drop followed by a flat tail.
"""),
            code("""
tab = pd.DataFrame({"complexity": [c.complexity for c in res.front],
                    "loss": [c.loss for c in res.front],
                    "score": scores(res.front),
                    "expression": [c.expression(["n"]) for c in res.front]})
print(tab.to_string(index=False))
best = select(res.front, "best")
print("\\nselected:", best.expression(["n"]))
print(form_check(to_sympy(best.tree), LAWS["bohr"]))
"""),
            md("""
`form_check` asks whether the selected form can represent the exact law on a
grid far beyond the data, and whether its fitted predictions there are within
5 %. With 1 % noise and 16 points, other forms fit the data as well as the
law does; which one is selected is then a property of the data, not of the
search.

## 4 · The vocabulary is a prior

Planck's law $B = x^3/(e^x - 1)$ is seven nodes with `exp`. Without `exp` it
cannot be written at all, and the best any search can do is a curve that
matches inside the data and leaves it outside. This is the repo's H6.
"""),
            code("""
law = LAWS["planck"]
x, y, w, _ = make_data(law, 24, 1e-3, 3)
no_exp = OPS.without("exp", "log")
r = exhaustive_search(x[:, None], y, w, ops=no_exp, max_size=6,
                      stop_loss=noise_floor(1e-3, 24))
best_no_exp = select(r.front)
planck_form = bi("/", un("cube", var(0)), bi("-", un("exp", var(0)), const()))
with_exp, l2 = fit_constants(planck_form, x[:, None], y, w)
xs = np.geomspace(0.05, 15, 300)[:, None]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.axvspan(0.5, 8, color=P.GRID, alpha=0.6, lw=0)
ax.loglog(xs, law.truth(xs[:, 0]), "--", color=P.INK_MUTED, label="Planck (exact)")
ax.loglog(xs, np.abs(evaluate(best_no_exp.tree, xs)), color="#eb6834",
          label=f"best without exp, size <= 6: {best_no_exp.expression()}")
ax.loglog(xs, evaluate(with_exp, xs), color="#2a78d6", lw=1.2, label="the 7-node form with exp")
ax.set_ylim(1e-4, 10); ax.legend(fontsize=7); ax.set_xlabel("x"); plt.show()
print("loss without exp:", best_no_exp.loss, "  with exp:", l2)
"""),
            md("""
The committed study runs the GP, exhaustive search and PySR with and without
`exp` on the same data, over the reporting seeds:
"""),
            code("""
v = load_table("symbolic", "vocabulary")
print(v[["vocabulary", "method", "seed", "recovered", "form_err", "extrap_dev",
         "expression"]].to_string(index=False))
"""),
            md("""
## 5 · SINDy: sparse regression for dynamics

For a trajectory, SINDy fixes a library of terms
$\\Theta(X) = [1, \\theta, \\omega, \\theta^2, \\ldots, \\sin\\theta, \\cos\\theta]$ and
solves $\\dot X = \\Theta(X)\\,\\Xi$ with sequentially thresholded least squares.
The library plays the role of the operator set. $\\dot X$ is not measured:
it has to be estimated, and a finite difference divides the noise by the
step.
"""),
            code("""
sys_ = sindy.pendulum()
t, X = sindy.simulate(sys_)
Xn = sindy.add_noise(X, 0.01, np.random.default_rng(3))
dt = t[1] - t[0]
_, d_fd = sindy.derivative(Xn, dt, "fd")
_, d_sg = sindy.derivative(Xn, dt, "savgol", window=51)
exact = np.array([sys_.rhs(0, s) for s in X])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t, d_fd[:, 0], color="#eb6834", lw=0.8, label="central differences")
ax.plot(t, d_sg[:, 0], color="#2a78d6", lw=1.2, label="Savitzky-Golay")
ax.plot(t, exact[:, 0], "--", color=P.INK_MUTED, label="exact")
ax.set_xlim(0, 3); ax.set_xlabel("t"); ax.set_ylabel("dtheta/dt"); ax.legend(); plt.show()
# threshold and window as chosen on the tuning seeds by the committed study
chosen = {c["method"]: c for c in load_json("symbolic", "sindy_chosen")["chosen"]
          if c["system"] == "pendulum"}
for method in ("fd", "savgol"):
    c = chosen[method]
    m = sindy.fit_sindy(t, Xn, sys_.state, threshold=c["threshold"], method=method,
                        window=int(c["window"]) or 21, trig=True)
    print(method, sindy.score_model(m, sys_))
    print("   ", "\\n    ".join(m.equations()))
"""),
            md("""
The threshold and window are the ones the committed study chose on seeds
3/7/19; its results on seeds 11/23/42 are in `results/symbolic/sindy_recovery.csv`.

The printed coefficients show the threshold's dilemma. The pendulum's damping
term is small (b = 0.2, against g/L = 9.8 for the restoring term). A
threshold low enough to keep it also keeps spurious terms that noise in the
states creates; one high enough to reject those can drop it. STLSQ has one
threshold for all terms, so a law with terms of very different sizes is the
hard case for it.

## 6 · AI Feynman, in one line

Udrescu & Tegmark (2020) attack the same problem differently: they fit a
neural network first, then use it to test for symmetries and separability
(does $f(x,y)$ split into $g(x)+h(y)$?), and recursively break the problem
into smaller ones before any brute-force search. It trades the tree search
for tests on a smooth surrogate.

## What to take away

- SR searches a space that grows geometrically; every practical method is a
  heuristic over it, and all of them share the Pareto front as their output.
- The formula it returns is chosen by a rule (the score), and with noise
  several forms fit equally well.
- Nothing outside the operator set can be found. The vocabulary is a prior,
  and should be stated wherever SR results are.
"""),
        ],
        "T11 · How symbolic regression works",
    )
