"""Build the notebooks -- one per physics problem, plus an overview.

`physprior notebooks [--execute] [name]`

Generated rather than hand-edited, for the same reason the numbers are: a
notebook written by hand drifts from the results it claims to show. Every
figure and number is computed from `results/` at execution time.

Animations are referenced with <img src="../figures/..."> rather than embedded,
so the notebooks stay small; they render in Jupyter opened from the project
root. Run `physprior figures` (or `make figures`) first if a GIF is
missing.
"""

from __future__ import annotations

import sys

import nbformat as nbf

from physprior.config import get_settings

HEADER = """import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import HTML, display
from physprior.viz import plots as P
from physprior.io import load_table, load_json
from physprior.config import get_settings
from physprior.benchmark.protocol import fit_arm

RESULTS = get_settings().results_dir
P.use_style()
pd.set_option("display.width", 160, "display.max_columns", 60)

def gif(path, width=560, caption=""):
    "Show an animation from figures/ without embedding it in the notebook."
    cap = f"<div style='color:#52514e;font-size:90%'>{caption}</div>" if caption else ""
    return HTML(f"<img src='../figures/{path}' width='{width}'>{cap}")
"""


# ---------------------------------------------------------------------------
# Every problem notebook follows the same four movements, in this order:
#
#   1  THE PROBLEM   the physical question, the governing law as maths, and a
#                    picture of the system before any fitting happens
#   2  THE DATA      where the numbers come from -- created by simulation, or
#                    pulled from an archive with its provenance
#   3  THE METHOD    what is actually being solved, and by which arms. The two
#                    jobs are not the same: DISCOVER a law you were not given
#                    (symbolic regression) or RECOVER a constant inside a law
#                    you were (physics fit, PINN)
#   4  THE RESULTS   plots, then a conclusion generated from results/
#
# A reader should be able to stop after section 1 and still know what question
# is being asked. `tests/test_notebooks.py` checks the order.
# ---------------------------------------------------------------------------

SECTIONS = ("1 · The problem", "2 · The data", "3 · The method", "4 · The results")


def section(n: int, subtitle: str = "") -> str:
    """The canonical heading for movement `n`, so the four are never renamed
    into something that only looks like the same structure."""
    tail = f" — {subtitle}" if subtitle else ""
    return f"## {SECTIONS[n - 1]}{tail}"


def method_block(*, discovers: bool, recovers: bool, arms: str, note: str = "") -> str:
    """Say plainly which of the two jobs this track is doing.

    Conflating them is the most common way to misread a physics-ML result: a
    method that recovers a constant inside a law it was handed has not
    discovered the law, and one that discovers a law has usually not measured
    anything to a useful precision.
    """
    jobs = []
    if discovers:
        jobs.append(
            "**Discovery** — find a law nobody supplied. Only `sr` (symbolic "
            "regression) can do this; it is given numbers and an operator set, "
            "never an equation."
        )
    if recovers:
        jobs.append(
            "**Recovery** — measure a constant *inside* a law that is supplied. "
            "`physics` fits it classically with a covariance; `pinn` fits it "
            "with a neural correction alongside, and `w_phys` sets how much "
            "correction is allowed."
        )
    body = "\n\n".join(f"{i + 1}. {j}" for i, j in enumerate(jobs))
    return (
        f"{section(3)}\n\n"
        f"Two different jobs travel under the name *physics-informed*, and this "
        f"track does the following:\n\n{body}\n\n"
        f"**Arms run here:** {arms}\n\n{note}"
    )


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def _nb(cells, title):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "physprior",
            "language": "python",
            "name": "physprior",
        },
        "language_info": {"name": "python"},
        "title": title,
    }
    return nb


# ---------------------------------------------------------------------------


def conclusion_cells(scope: str):
    """The closing section every notebook ends in.

    Generated, not written: the verdict table, the prose and the
    falsification list are all rendered from `results/` by
    `physprior.reporting.conclusions`, so a notebook cannot keep claiming a
    win after the numbers stop supporting it. A question whose arms are
    separated by less than the seed-to-seed spread is reported as a tie, and
    if nothing is decisive the conclusion says exactly that.
    """
    return [
        md("""
---

## Conclusion

Everything below is rendered from `results/` at execution time by
`physprior.reporting.conclusions` — the verdicts, the margins and the prose.
A gap smaller than the seed-to-seed spread on the reporting seeds
(11 / 23 / 42) is reported as a **tie**, not rounded into a win.
"""),
        code(f"""
from physprior.reporting import conclusions as C
verdicts = C.verdict_frame({scope!r})
display(verdicts)
"""),
        code(
            """
from IPython.display import Markdown
display(Markdown(C.conclusion_markdown(%r)))
"""
            % scope
        ),
        code(
            r"""
lines = ["- " + line for line in C.what_would_change_this(%r)]
display(Markdown("**What would change this conclusion**\n\n" + "\n".join(lines)))
"""
            % scope
        ),
    ]


def nb_overview():
    return _nb(
        [
            md("""
# physprior — what does a physics prior buy you?

A neural network can fit almost any curve. A physicist writes a law with two
constants in it. On **real measured data** — LIGO, COBE/FIRAS, NIST, JPL —
and on **simulations where the answer is known exactly**, which should you
use, and for what?

This notebook is the map. Each claim links to the notebook that measures it.
"""),
            code(HEADER),
            md("""
## The question, made concrete

Five arms run on every track, through the same six protocol questions, so
that *physics-informed* is compared against something rather than asserted.

| arm | knows the law? | returns a constant? | returns a formula? |
|---|---|---|---|
| `oracle` | published law **and** constants — a ceiling, not a competitor | — | yes |
| `physics` | the law, constants fitted | yes, with a covariance | yes |
| `pinn` | the law **plus** a learned correction | yes | yes + correction |
| `sr` | **no** — searches for a law | sometimes | whatever it finds |
| `nn` | no — a tuned MLP | no | no |

The `pinn` arm comes in two shapes, and which one a track uses follows from
whether its law is a differential equation or an algebraic relation:
"""),
            code("""
P.fig_pinn_anatomy(); plt.show()
"""),
            md("""
## The answer, in one figure

The honest scorecard across the four real-data tracks: **most protocol
questions do not separate the arms at all**, and the `pinn` arm wins one cell
in twelve.
"""),
            code("""
from physprior.reporting import conclusions as C
frame = C.verdict_frame("all")
counts = frame.verdict.str.split(" ").str[0].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
ax = axes[0]
ax.barh(list(counts.index), list(counts.values),
        color=[{"WIN": P.ARM_COLOR["pinn"]}.get(k, P.INK_MUTED) for k in counts.index],
        edgecolor=P.SURFACE, linewidth=2, height=0.55)
for k, v in counts.items():
    ax.annotate(f"  {v}", xy=(v, k), va="center", fontsize=9, color=P.INK_2)
ax.set_xlabel("protocol questions"); ax.grid(axis="y", visible=False)
ax.set_title("Most questions are ties")

ax = axes[1]
wins = frame[frame.verdict.str.startswith("WIN")].winner.value_counts()
ax.barh(list(wins.index), list(wins.values),
        color=[P.ARM_COLOR.get(a, P.INK_MUTED) for a in wins.index],
        edgecolor=P.SURFACE, linewidth=2, height=0.55)
for k, v in wins.items():
    ax.annotate(f"  {v}", xy=(v, k), va="center", fontsize=9, color=P.INK_2)
ax.set_xlabel("decisive wins"); ax.grid(axis="y", visible=False)
ax.set_title("and who takes the rest")
plt.show()
"""),
            md("""
That is a real result, and it is not the one the field usually reports. It is
also **not** evidence that physics-informed learning does not work — it is
evidence that these four tracks are mostly the wrong test.

## Three regimes, and only one of them is a fair test

| regime | example here | who wins |
|---|---|---|
| the law is **exact** | `gravity/kepler` | `physics` — the prior has nothing to add |
| the missing piece is **degenerate** with the law | `quantum/cmb` | nobody — the constant is not identifiable |
| the missing piece is **distinguishable** | [T7](tutorials/T7_when_the_prior_wins.ipynb) | **`pinn`, by an order of magnitude** |

The third regime is the normal condition of applied physics, and it was
missing from the real tracks. [T7](tutorials/T7_when_the_prior_wins.ipynb)
builds it as a controlled experiment:
"""),
            code("""
import pandas as pd
eps = pd.read_csv("results/neglected_eps.csv")
P.fig_neglected_sweep(eps, "eps", "size of the term left out of the law  (eps)",
                      "Given a law that is incomplete in a way it can see")
plt.show()
"""),
            md("""
## The constants this project recovered from real data
"""),
            code("""
h = json.load(open(RESULTS / "headline.json"))
P.fig_parameter_recovery(h["parameter_recovery"]); plt.show()
"""),
            md("""
## Extrapolation, across the benchmark tracks

Each arm is fitted on the *low* part of its range and asked about the *high*
part. This is where a prior either pays or does not.
"""),
            code("""
P.fig_cross_track_extrapolation(h["extrapolation_summary"]); plt.show()
"""),
            md("""
## The findings

1. **Inside the training range, with enough clean data, a tuned black box is
   competitive.** Physics buys little there, and claims to the contrary
   usually compare against an untuned baseline.
2. **Outside it the gap is orders of magnitude — but the mechanism is
   identifiability, not the mere presence of a law.** On `quantum/cmb`
   out-of-band error falls ~56× as the fitted band reaches the Rayleigh–Jeans
   regime *while in-band error gets worse*.
3. **Only the physics arms return something a physicist can argue with** —
   QED in hydrogen, the post-Newtonian expansion in GW150914, and a
   truncation error masquerading as a 56σ refutation of general relativity.
4. **The network eats the physics if you let it.** At `w_phys = 0` the
   recovered `GM_sun` is 19.5% wrong while held-out error barely moves.
5. **A pipeline you have not injected into has no error budget.**

### Where to go next

| | |
|---|---|
| [tutorials/](tutorials/README.md) | a step-by-step PINN course, T1 → T7 |
| [01_gravity](01_gravity.ipynb) · [02_relativity](02_relativity.ipynb) · [03_quantum](03_quantum.ipynb) | the three problems in full |
| [T7](tutorials/T7_when_the_prior_wins.ipynb) | the regime where the prior wins, and why the others do not |
"""),
            *conclusion_cells("all"),
        ],
        "physprior overview",
    )


def nb_gravity():
    return _nb(
        [
            md("""
# Problem: gravity

Newtonian orbital mechanics — the case where the law is **exact**, so
whatever a method fails to recover is the method's own error.
"""),
            code(HEADER),
            # ---------------- 1 · the problem -------------------------
            md(f"""
{section(1, "what is being asked")}

Two bodies attract along the line between them:

$$\\ddot{{\\mathbf{{r}}}} = -\\frac{{GM}}{{r^{{3}}}}\\,\\mathbf{{r}}$$

and from that follows **Kepler's third law**, which relates a planet's
orbital period to the size of its orbit through a single constant:

$$P = 2\\pi\\sqrt{{\\frac{{a^{{3}}}}{{GM}}}}$$

Three questions, and they are not the same question:

1. Can a method **discover** the exponent $-2$ in the force law, never having
   been told it?
2. Can a method **recover** $GM_{{\\odot}}$ from real planetary data, and to
   what precision?
3. Does knowing the law help **outside** the range the data covers?

Before any fitting, here is the system itself — the Sun and Mercury, in the
centre-of-mass frame, so the Sun moves too:
"""),
            code("""
display(gif("gravity/two_body_mercury.gif", 460,
            "e = 0.206 makes perihelion the hard part."))
"""),
            # ---------------- 2 · the data ----------------------------
            md(f"""
{section(2, "created, then pulled")}

This track uses **both** kinds, and the order matters: the simulation is the
control that the real-data number is read against.

### 2a · Created — and the integrator is part of the physics model

A simulation is only a control if its own error is smaller than the effect
being studied. Velocity Verlet is *symplectic*: its energy error oscillates
and stays bounded. RK4 has the better local error and is not symplectic, so
its error **drifts** and a bound orbit slowly unbinds.
"""),
            code("""
ic = load_table("gravity", "integrator_comparison")
fig, ax = plt.subplots(figsize=(6.8,4.0))
ax.loglog(ic.dt_days, ic.verlet_energy_drift, "o-", color=P.ARM_COLOR["physics"],
          label="velocity Verlet (symplectic)")
ax.loglog(ic.dt_days, ic.rk4_energy_drift, "o-", color=P.ARM_COLOR["nn"],
          label="RK4 (not symplectic)")
ax.set_xlabel("step (days)"); ax.set_ylabel("relative energy drift over 200 years")
ax.set_title("Why the integrator is a physics choice"); ax.legend()
plt.show()
worst = ic.iloc[ic.dt_days.idxmax()]
print(f"at {worst.dt_days:g}-day steps: Verlet {worst.verlet_energy_drift:.1%}, "
      f"RK4 {worst.rk4_energy_drift:.0%}")
"""),
            md("""
With the integrator chosen, the simulations are generated: a periodic
three-body choreography, the same configuration perturbed by one part in
10⁹, and the eight planets from **real JPL initial conditions**.
"""),
            code("""
display(gif("gravity/three_body_figure8.gif", 420, "the periodic choreography"))
display(gif("gravity/three_body_chaotic.gif", 420, "perturbed by 1e-3"))
"""),
            code("""
m = load_json("gravity", "problem_meta")
ly = m["simulations"]["lyapunov"]
print(f"Lyapunov exponent  lambda = {ly['lambda']:.4f} per time unit")
print(f"e-folding time            = {ly['e_folding_periods']:.2f} periods")
display(gif("gravity/lyapunov.png", 560))
"""),
            code("""
display(gif("gravity/solar_system_inner.gif", 460,
            "Only the state at the epoch comes from the ephemeris; "
            "everything after is this integrator's own Newtonian solution."))
print("energy drift over the run:", m["simulations"]["solar_system_drift"])
"""),
            md("""
### 2b · Pulled — JPL DE441

Eight planets, from the JPL Horizons API. The loader records the URL, byte
count and SHA-256 of every file it read, asserts the units it promises, and
never edits the download.
"""),
            code("""
from physprior.problems.gravity import kepler
prob, meta = kepler.problem()
meta = {**meta, **load_json("gravity/kepler", "meta")}
# provenance is a list when a track read several files, a dict when one.
prov = meta["provenance"]
for rec in (prov if isinstance(prov, list) else [prov]):
    print(f"source : {rec.get('file', rec.get('note',''))}  "
          f"{rec.get('bytes','?')} bytes  sha256 {str(rec.get('sha256',''))[:12]}...")
print(f"planets  : {len(prob)}   a = {prob.x[:,0].min():.3f} to {prob.x[:,0].max():.3f} AU")
print(f"published GM_sun = {meta['published_GM_sun']:.6e} m^3/s^2")

fig, ax = plt.subplots(figsize=(6.4,3.8))
ax.loglog(prob.x[:,0], prob.y, "o", markersize=9, color=P.ARM_COLOR["physics"],
          markeredgecolor=P.SURFACE, markeredgewidth=1.4)
for a_au, per, name in zip(prob.x[:,0], prob.y, meta.get("planets", []) or []):
    ax.annotate(f"  {name}", xy=(a_au, per), fontsize=8, color=P.INK_2, va="center")
ax.set_xlabel("semi-major axis a (AU)"); ax.set_ylabel("period P (days)")
ax.set_title("The data: eight points, five decades of nothing else")
plt.show()
"""),
            # ---------------- 3 · the method --------------------------
            md(
                method_block(
                    discovers=True,
                    recovers=True,
                    arms="`oracle` · `physics` · `pinn` · `sr` · `nn`",
                    note=(
                        "Both jobs run here, on different data. Discovery runs "
                        "on the **simulation**, where the answer is known "
                        "exactly, so the number it returns is the method's own "
                        "noise floor. Recovery runs on the **real ephemeris**, "
                        "and its residual is read against that floor.\n\n"
                        "Because Kepler's law is an algebraic relation rather "
                        "than a differential equation, the `pinn` arm here is "
                        "the **law + correction** form — panel B below, not the "
                        "residual form."
                    ),
                )
            ),
            code("""
P.fig_pinn_anatomy(); plt.show()
"""),
            # ---------------- 4 · the results -------------------------
            md(f"""
{section(4, "discovery first, on the simulation")}

Symbolic regression is handed a simulated orbit's radius and acceleration —
no equation, no template, no units — and asked what relates them.
"""),
            code("""
fl = m["discovery"]["force_law"]
print("SR expression      :", fl["sr_expression"])
print(f"exponent, SR       : {fl['exponent_sr']:.7f}   (true -2)")
print(f"exponent, log-log  : {fl['exponent_loglog']:.7f}")
print(f"GM, direct fit     : {fl['mu_direct_rel_error_ppb']:+.1f} ppb")
print(f"GM, log-log interc.: {fl['mu_loglog_rel_error_ppm']:+.1f} ppm")
"""),
            md("""
**That is the floor.** An exponent of −1.9999969 and `GM` to 0.6 ppb is what
this machinery can do when the law is exactly what was put in. Any larger
residual on real data is physics, not method.

The same law again, from the *chaotic* three-body run — chaos destroys
predictability, not the law generating it:
"""),
            code("""
cl = m["discovery"]["chaotic_law"]
print(f"G*m recovered : {cl['Gm_recovered']:.9f}  (true 1)  -> {cl['Gm_rel_error_ppm']:+.3f} ppm")
print(f"fit residual  : {cl['residual_rel']:.2e}   over {cl['n_samples']:,} samples")
print(f"Lyapunov      : {cl['lyapunov']:.4f}  (the trajectory really is chaotic)")
"""),
            code("""
kl = m["discovery"]["kepler_law"]
display(load_table("gravity","kepler_from_simulation"))
print("SR:", kl["sr_expression"])
print(f"exponent: SR {kl['exponent_sr']:.6f} | log-log {kl['exponent_loglog']:.6f}  (true 1.5)")
"""),
            md("""
### Recovery, watched live

The PINN's physical constant is an ordinary trainable parameter. Here it is
walking toward its published value while the fit tightens:
"""),
            code("""
tr = m["training"]
display(gif("gravity/pinn_learning_orbit.gif", 820))
print(f"GM started at {tr['GM_initial']:.4e}")
print(f"GM ended at   {tr['GM_recovered']:.4e}   (true {tr['GM_true']:.4e})")
print(f"final error   {tr['GM_rel_error_ppm']:+.0f} ppm, from data with "
      f"{tr['noise_frac']*100:.0f}% noise")
"""),
            md("### Recovery on the real ephemeris — all five arms"),
            code("""
fits = {a: fit_arm(a, prob, np.arange(len(prob)), seed=11)
        for a in ["oracle","physics","pinn","sr","nn"]}
P.fig_overview(prob, fits, logx=True, logy=True,
               title="Kepler's third law: eight planets, five arms")
plt.show()
"""),
            md("""
### Does the law help outside the data?

Train on the four terrestrial planets; predict the four giants. This is a 20×
extrapolation in `a`, and it is where a physics prior is supposed to earn its
keep.
"""),
            code("""
ex = load_table("gravity/kepler","extrapolation")
P.fig_extrapolation_bars(ex, "Train on the four terrestrial planets, predict the four giants")
plt.show()
"""),
            md("""
### And how much physics belongs in the loss?

`w_phys` is the dial. Watch the recovered constant, not the fit quality.
"""),
            code("""
wp = load_table("gravity/kepler","sweep_physics_weight")
P.fig_physics_weight(wp, "gravity/kepler", param_key="GM",
                     published=meta["published_GM_sun"])
plt.show()
"""),
            *conclusion_cells("gravity"),
        ],
        "Problem: gravity",
    )


def nb_relativity():
    return _nb(
        [
            md("""
# Problem: relativity

The case where the law is **an approximation you can truncate at the wrong
order** — and where a plausible numerical choice once produced a 56σ
refutation of general relativity.
"""),
            code(HEADER),
            # ---------------- 1 · the problem -------------------------
            md(f"""
{section(1, "two laws that are almost right")}

**Mercury.** Newton's orbit closes. Schwarzschild's does not: the relativistic
orbit equation carries one extra term,

$$\\frac{{d^{{2}}u}}{{d\\varphi^{{2}}}} + u = \\frac{{GM}}{{h^{{2}}}}
+ \\underbrace{{\\frac{{3GM}}{{c^{{2}}}}u^{{2}}}}_{{\\text{{general relativity}}}}$$

and that term advances the perihelion by 43 arcsec/century. It is **8×10⁻⁸**
of Mercury's acceleration, which makes this a numerics problem before it is a
physics problem.

**GW150914.** A binary inspiral radiates, so its frequency sweeps upward:

$$\\dot{{f}} = \\frac{{96}}{{5}}\\pi^{{8/3}}
\\left(\\frac{{G\\mathcal{{M}}}}{{c^{{3}}}}\\right)^{{5/3}} f^{{11/3}}
\\left[1 + \\text{{1PN}} + \\text{{1.5PN}} + \\text{{2PN}} + \\cdots\\right]$$

That bracket is a **series**, and you choose where to stop. The question this
track exists to answer: *does anything in the fit tell you that you stopped
too early?*
"""),
            # ---------------- 2 · the data ----------------------------
            md(f"""
{section(2, "created, then pulled")}

### 2a · Created — the precession, from the equation above

Integrated twice, with the GR term on and off. The difference is the
precession.
"""),
            code("""
m = load_json("relativity", "problem_meta")
mp = m["simulations"]["mercury_precession"]
print(f"simulated GR precession : {mp['precession_arcsec_per_century']:.4f} arcsec/century")
print(f"analytic 6 pi GM/(c^2 a(1-e^2)) : {mp['analytic_arcsec_per_century']:.4f}")
print(f"agreement : {abs(mp['precession_arcsec_per_century']-mp['analytic_arcsec_per_century']):.2e} arcsec")
"""),
            code("""
display(gif("relativity/schwarzschild_precession.gif", 460))
ld = m["simulations"]["light_deflection"]
print("Light bending at the solar limb:")
print(f"  general relativity, 4GM/c^2b : {ld['deflection_gr_arcsec']:.4f} arcsec")
print(f"  Newtonian half-value          : {ld['deflection_newtonian_arcsec']:.4f} arcsec")
"""),
            md("""
The perihelion is located by **root-finding `du/dφ`**, not by fitting a
parabola to a sampled grid. The shift is 5e-7 rad per orbit and a grid
estimate is good to ~1e-6 — bigger than the effect. The first version of this
simulation duly reported a Newtonian "precession" 20% larger than the GR one.

### 2b · Pulled — LIGO strain, and Mercury's ephemeris
"""),
            code("""
from physprior.data.sources import gwosc as gw
tr = gw.frequency_track()
fig, axes = plt.subplots(2,1, figsize=(7.6,5.6), sharex=True)
w = (tr.strain_t>-0.16)&(tr.strain_t<0.06)
axes[0].plot(tr.strain_t[w], tr.strain_h[w], color=P.ARM_COLOR["physics"], lw=1.2)
axes[0].set_ylabel("whitened strain"); axes[0].set_title("GW150914, H1+L1 combined")
axes[1].plot(tr.t_s, tr.f_hz, "o-", color=P.ARM_COLOR["pinn"], markersize=6)
axes[1].set_xlabel("time from merger (s)"); axes[1].set_ylabel("frequency (Hz)")
axes[1].set_title("The seven usable cycles -- this is the whole dataset")
plt.show()
print(f"{len(tr.t_s)} points. That is not a typo: the event's SNR of 24 is "
      "accumulated coherently over the waveform, and a per-cycle frequency "
      "needs per-cycle SNR.")
"""),
            # ---------------- 3 · the method --------------------------
            md(
                method_block(
                    discovers=False,
                    recovers=True,
                    arms="`oracle` · `physics` · `pinn` · `sr` · `nn`",
                    note=(
                        "**No discovery here.** Seven points cannot support a "
                        "search over functional forms, so `sr` runs as a "
                        "baseline rather than as a discovery method. The job is "
                        "recovery: the chirp mass from GW150914, and the GR "
                        "coefficient α from Mercury, which general relativity "
                        "says is exactly 1.\n\n"
                        "This track's law **is a differential equation**, so "
                        "its `pinn` arm is the residual PINN of Raissi et al. "
                        "— panel A — not the law-plus-correction form the other "
                        "tracks use. The chirp mass is an `nn.Parameter` inside "
                        "the residual, so it receives a gradient through the "
                        "physics term."
                    ),
                )
            ),
            code("""
P.fig_pinn_anatomy(); plt.show()
"""),
            # ---------------- 4 · the results -------------------------
            md(f"""
{section(4, "does the fit notice a truncated law?")}

Same data, same code, same fitting — only the PN order changes.
"""),
            code("""
gwm = load_json("relativity/gw150914", "meta")
ab = pd.DataFrame(gwm["pn_ablation"])
display(gif("relativity/gw150914/pn_ablation.png", 600))
ok = ab[ab.converged]
print(f"chirp mass spread across orders : {ok.Mc.max()-ok.Mc.min():.2f} Msun")
print(f"RMSE spread across orders       : {ok.rmse_hz.max()-ok.rmse_hz.min():.3f} Hz")
print("")
print("the 1PN row pinned to its bound and is excluded -- a parameter at its")
print("bound has not converged, whatever the optimiser reports")
"""),
            md("""
**Goodness of fit does not diagnose a wrong law.** At Newtonian order the
chirp mass is biased by ~+9 M☉ with a formal error of 4.1 — a confident wrong
answer — while the RMSE hardly moves.

### Is that bias real, or is it the pipeline's?

The only way to find out is to put a **known** answer through the identical
code.
"""),
            code("""
inj = m["discovery"]["injection"]
print(f"injected Mc = {inj['mchirp_true']:.2f} M_sun, "
      f"{inj['n_usable']}/{inj['n_trials']} usable, "
      f"SNR threshold {inj['snr_threshold']}")
for k in ("pn0","pn3"):
    if k in inj:
        print(f"  {k}: injection bias {inj[k]['bias_pct']:+.1f}%")
"""),
            code("""
tc = load_table("relativity","threshold_calibration")
fig, ax = plt.subplots(figsize=(6.6,3.8))
ax.axhline(0, color=P.INK_2, lw=1.4)
ax.plot(tc.snr_threshold, tc.bias_pct, "o-", color=P.ARM_COLOR["pinn"], markersize=9,
        markeredgecolor=P.SURFACE, markeredgewidth=1.4)
ax.set_xlabel("envelope SNR a cycle must clear"); ax.set_ylabel("bias in recovered Mc (%)")
ax.set_title("The a-priori choice of 2.0 biased the answer by -70%")
plt.show()
print("the value adopted is the smallest whose injection bias is under 1%,")
print("chosen on injections and never on the real event")
"""),
            md("""
### Mercury: the result is α once it has stopped moving
"""),
            code("""
gm = load_json("relativity/mercury", "meta")
P.fig_alpha_convergence(gm["gr_convergence"]); plt.show()
gr = gm["gr"]
print(f"converged alpha = {gr['alpha_GR']:.6f} +- {gr['alpha_sigma']:.6f}   (Einstein: 1)")
"""),
            *conclusion_cells("relativity"),
        ],
        "Problem: relativity",
    )


def nb_quantum():
    return _nb(
        [
            md("""
# Problem: quantum

Two tracks that fail in **opposite** ways: hydrogen, where the law is so
nearly exact that *its own failure* is visible, and the CMB, where the law is
transcendental and the observed band cannot identify it.
"""),
            code(HEADER),
            # ---------------- 1 · the problem -------------------------
            md(f"""
{section(1, "one equation, two very different tracks")}

Everything here comes from the time-independent Schrödinger equation,

$$-\\tfrac{{1}}{{2}}\\psi''(x) + V(x)\\,\\psi(x) = E\\,\\psi(x)$$

**Hydrogen.** Bohr's law says the levels go as $-1/n^{{2}}$:

$$E_{{n}} = -\\frac{{R_{{H}}}}{{n^{{2}}}}$$

It is very nearly right, and the interesting question is *how* it is wrong —
relativistic and QED corrections shift the 1s level by about 10 ppm, and NIST
measures the levels far more precisely than that.

**The CMB.** Planck's law is transcendental:

$$B_{{\\nu}}(T) = \\frac{{2h\\nu^{{3}}}}{{c^{{2}}}}
\\frac{{1}}{{e^{{h\\nu/kT}} - 1}}$$

Over the band FIRAS actually observed, $x = h\\nu/kT$ runs from 1.2 to 11.3 —
almost all Wien — and there $e^{{-x}}$ and $1/(e^{{x}}-1)$ are nearly the same
function. **The denominator is not identifiable from the data.** That is the
whole track.
"""),
            # ---------------- 2 · the data ----------------------------
            md(f"""
{section(2, "created, then pulled")}

### 2a · Created — and the solver's error is measured, not assumed
"""),
            code("""
m = load_json("quantum", "problem_meta")
sp = m["simulations"]["spectra"]
rows = [{"system": k, "law": v["law"], "max rel error": v["max_rel_error"],
         "E_0": v["energy"][0], "E_0 exact": v["exact"][0]} for k,v in sp.items()]
display(pd.DataFrame(rows))
"""),
            code("""
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
for ax, name in zip(axes, ["infinite_well","harmonic","hydrogen"]):
    display(gif(f"quantum/eigenstates_{name}.png", 340))
plt.close(fig)
"""),
            md("""
**Richardson extrapolation, because the solver was coarser than the effect.**
Plain second-order differences give the hydrogen levels to ~300 ppm. The QED
shift is 10.8 ppm. A solver 30× less accurate than the effect cannot see it,
and differencing anyway would report discretisation error as physics.
"""),
            code("""
gc = pd.DataFrame(m["simulations"]["grid_convergence"])
fig, ax = plt.subplots(figsize=(6.4,3.8))
ax.loglog(gc.dx, gc.max_rel_error.abs(), "o-", color=P.ARM_COLOR["pinn"], markersize=8,
          markeredgecolor=P.SURFACE, markeredgewidth=1.3)
ax.set_xlabel("grid spacing dx"); ax.set_ylabel("|max relative error|")
ax.set_title("The solver's own error, measured")
plt.show()
display(pd.DataFrame(m["simulations"]["r_min_sweep"]))
print("the r_min sweep is the companion warning: refine dx all you like,")
print("the inner boundary sets the floor")
"""),
            code("""
tu = m["simulations"]["tunnelling"]
display(gif("quantum/tunnelling.gif", 640))
print(f"packet energy   : {tu['energy']:.3f}")
print(f"barrier height  : {tu['barrier_height']:.3f}")
print(f"norm conserved to {tu['norm_drift']:.1e} -- unitary by construction")
"""),
            md("""
### 2b · Pulled — NIST hydrogen levels, and the COBE/FIRAS monopole
"""),
            code("""
from physprior.problems.quantum import cmb, hydrogen
prob, hmeta = hydrogen.problem()
hmeta = {**hmeta, **load_json("quantum/hydrogen", "meta")}
cprob, cmeta = cmb.problem()
cmeta = {**cmeta, **load_json("quantum/cmb", "meta")}
print(f"NIST H I : {len(prob)} levels")
print(f"FIRAS    : {len(cprob)} channels, {cmeta['nu_ghz_range'][0]:.0f}-"
      f"{cmeta['nu_ghz_range'][1]:.0f} GHz  (turnover at {cmeta['turnover_ghz']:.0f})")
print(f"caveat   : {cmeta['caveat']}")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
axes[0].plot(prob.x[:,0], prob.y, "o", color=P.ARM_COLOR["physics"], markersize=7,
             markeredgecolor=P.SURFACE, markeredgewidth=1.2)
axes[0].set_xlabel("n"); axes[0].set_ylabel("level (cm^-1)"); axes[0].set_title("NIST hydrogen")
axes[1].plot(cprob.x[:,0], cprob.y, "o", color=P.ARM_COLOR["pinn"], markersize=5,
             markeredgecolor=P.SURFACE, markeredgewidth=1.0)
axes[1].set_xlabel("frequency"); axes[1].set_ylabel("intensity")
axes[1].set_title("COBE/FIRAS monopole")
plt.show()
"""),
            # ---------------- 3 · the method --------------------------
            md(
                method_block(
                    discovers=True,
                    recovers=True,
                    arms="`oracle` · `physics` · `pinn` · `sr` · `nn`",
                    note=(
                        "Both jobs, and this is the track where the difference "
                        "bites hardest. Discovery runs on the **solved "
                        "spectra**, where the law is known, and succeeds: "
                        "symbolic regression finds `n^2`, `(n + 1/2)` and "
                        "`-1/n^2`. On **FIRAS** the same machinery fails to "
                        "find Planck's law and returns a Wien-like exponential "
                        "instead — and that negative result is reported at the "
                        "same size as the successes, with the control that "
                        "isolates its cause.\n\n"
                        "Both laws are algebraic, so both `pinn` arms are the "
                        "**law + correction** form — panel B."
                    ),
                )
            ),
            code("""
P.fig_pinn_anatomy(); plt.show()
"""),
            # ---------------- 4 · the results -------------------------
            md(f"""
{section(4, "discovery on the simulation, recovery on the real thing")}
"""),
            code("""
display(load_table("quantum","spectrum_laws"))
for r in m["discovery"]["spectrum_laws"]:
    print(f"[{r['system']:9s}] {r['law']}")
    print(f"    SR: {r['sr_expression']}")
"""),
            md("""
### Hydrogen: the law's own failure, from two directions
"""),
            code("""
f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
print(f"R fitted from the levels : {f.params['R']:.4f} cm^-1")
print(f"Bohr's prediction        : {hmeta['bohr_rydberg_H_icm']:.4f} cm^-1")
print(f"gap                      : {hmeta['limit_minus_bohr_ppm']:.2f} ppm  = QED + relativistic")
display(gif("quantum/hydrogen/bohr_residual.png", 600))
"""),
            code("""
sv = m["discovery"]["schrodinger_vs_nist"]
print(f"solver error        : {sv['solver_max_rel_error_ppm']:.1f} ppm "
      f"(plain finite differences: {sv['solver_without_richardson_ppm']:.0f} ppm)")
print(f"simulation vs NIST  : {sv['mean_gap_ppm']:+.1f} ppm (mean over n)")
print("the same QED shift, reached by solving the equation rather than fitting the law")
"""),
            md("""
### The CMB: where fitting well and finding the law come apart
"""),
            code("""
cfits = {a: fit_arm(a, cprob, np.arange(len(cprob)), seed=11)
         for a in ["oracle","physics","pinn","sr","nn"]}
P.fig_overview(cprob, cfits, title="COBE/FIRAS: every arm fits the band")
plt.show()
print("SR expression:", cfits["sr"].expression)
"""),
            code("""
ctrl = pd.DataFrame(json.load(open(RESULTS / "_band_control.json")))
display(gif("quantum/cmb/band_coverage_control.png", 600))
fig, ax = plt.subplots(figsize=(6.6,3.8))
ax.semilogy(ctrl.x_min, ctrl.max_rel_dev_in_band, "o-", color=P.ARM_COLOR["physics"],
            label="inside the fitted band")
ax.semilogy(ctrl.x_min, ctrl.max_rel_dev_outside, "o-", color=P.ARM_COLOR["sr"],
            label="outside it")
ax.set_xlabel("lowest x = h nu / kT reached by the band")
ax.set_ylabel("max relative deviation from Planck")
ax.set_title("In-band accuracy is anti-correlated with having found the law")
ax.legend(); plt.show()
"""),
            md("""
**Fit quality on the observed range is not evidence that you have discovered
anything.** The FIRAS-coverage run has the *best* in-band fit of the five and
one of the worst extrapolations.
"""),
            *conclusion_cells("quantum"),
        ],
        "Problem: quantum",
    )


BUILDERS = {
    "00_overview": nb_overview,
    "01_gravity": nb_gravity,
    "02_relativity": nb_relativity,
    "03_quantum": nb_quantum,
}


def _display(path, root):
    """A printable path, even when the output directory is outside the repo.

    `notebooks_dir` is env-overridable, so it is not always under `root` --
    and a bare `relative_to` raises there rather than printing.
    """
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def build(execute: bool = False, only: str | None = None) -> None:
    settings = get_settings()
    settings.notebooks_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        if only and only not in name:
            continue
        nb = fn()
        path = settings.notebooks_dir / f"{name}.ipynb"
        nbf.write(nb, path)
        print(f"wrote {_display(path, settings.root)}")
        if execute:
            from nbclient import NotebookClient

            print("  executing ...", flush=True)
            client = NotebookClient(
                nb,
                timeout=3600,
                kernel_name="physprior",
                resources={"metadata": {"path": str(settings.root)}},
                allow_errors=True,
            )
            client.execute()
            nbf.write(nb, path)
            errs = [
                c
                for c in nb.cells
                if any(o.get("output_type") == "error" for o in c.get("outputs", []))
            ]
            print(f"  done, {len(errs)} cell(s) with errors")


if __name__ == "__main__":
    build(
        execute="--execute" in sys.argv,
        only=next((a for a in sys.argv[1:] if not a.startswith("-")), None),
    )
