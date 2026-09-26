"""A step-by-step PINN course, generated as executable notebooks.

    physprior tutorials [--execute] [name]

Six notebooks, in order, each building on the last:

    T1  what a PINN is          a network that obeys an equation, not data
    T2  forward and inverse     solving vs recovering a constant
    T3  gravity                 Newtonian orbits; what a physics prior buys
    T4  relativity              when the law is a TRUNCATED expansion
    T5  quantum                 learning a wave function, with no data at all
    T6  when PINNs fail         the failure catalogue, measured in this repo

The arc is deliberate: the difficulty rises with the physics. Gravity's law is
exact and closed-form, relativity's is an expansion that can be truncated at
the wrong order, and quantum's is an eigenvalue problem with no data on the
right-hand side at all. Each notebook ends where the next one starts.

Generated rather than hand-written, for the same reason the results are: a
notebook written by hand drifts from the code it claims to teach.
"""

from __future__ import annotations

import sys

import nbformat as nbf

from physprior.config import get_settings
from physprior.reporting.notebooks import section

SETUP = """import warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, matplotlib.pyplot as plt
from physprior.viz import plots as P
P.use_style()
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__)
"""


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
# T1 -- what a PINN is
# ---------------------------------------------------------------------------


def t1_what_is_a_pinn():
    return _nb(
        [
            md("""
# T1 · What a physics-informed neural network actually is

A neural network is usually trained to match **data**. A PINN is trained to
satisfy an **equation**. That is the whole idea, and everything else is
consequence.

We will solve

$$\\frac{d^2x}{dt^2} = -\\omega^2 x, \\qquad x(0) = 1, \\quad \\dot{x}(0) = 0$$

whose answer is $x(t) = \\cos(\\omega t)$ — so we can check every step against
something known. **There is no training data anywhere in this notebook.**
"""),
            code(SETUP),
            md("""
## Step 1 · The network is the solution

Write $x_\\theta(t)$: a small network taking a time and returning a position.
Not a fit to samples of the solution — the network *is* the candidate
solution, and training will push it toward obeying the ODE.
"""),
            code("""
def mlp(width=32, depth=3):
    layers = [torch.nn.Linear(1, width), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))

torch.manual_seed(0)
net = mlp()
print(net)
print("parameters:", sum(p.numel() for p in net.parameters()))
"""),
            md("""
### Why `tanh` and never `ReLU`

The loss will differentiate the network's output **twice**. A ReLU network is
piecewise linear, so its second derivative is zero almost everywhere and the
physics term would be identically zero. This is not a style preference — a
ReLU PINN on a second-order equation silently trains on nothing.
"""),
            code("""
relu_net = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
t = torch.linspace(0, 1, 5, requires_grad=True)
for name, f in (("tanh", net), ("relu", relu_net)):
    y = f(t.unsqueeze(-1)).squeeze(-1)
    dy = torch.autograd.grad(y.sum(), t, create_graph=True)[0]
    d2y = torch.autograd.grad(dy.sum(), t, create_graph=True)[0]
    print(f"{name}: second derivative = {d2y.detach().numpy().round(6)}")
"""),
            md("""
## Step 2 · The initial conditions, as a hard constraint

Two ways to impose $x(0)=1,\\ \\dot x(0)=0$:

* **soft** — add $\\lambda\\,[(x_\\theta(0)-1)^2 + \\dot x_\\theta(0)^2]$ to the
  loss. Now there is a weight $\\lambda$ to tune, and the conditions hold only
  approximately.
* **hard** — build them into the architecture:

$$x_\\theta(t) = 1 + t^2\\,\\mathrm{NN}(t)$$

At $t=0$ this is exactly 1, and its derivative is exactly 0, for *any*
network. No weight, no approximation. **Prefer hard constraints wherever the
algebra allows them** — it is one fewer hyperparameter nobody can justify.
"""),
            code("""
def x_hard(t, f=None):
    f = net if f is None else f
    return 1.0 + t**2 * f(t.unsqueeze(-1)).squeeze(-1)

t0 = torch.zeros(1, requires_grad=True)
x0 = x_hard(t0)
v0 = torch.autograd.grad(x0.sum(), t0, create_graph=True)[0]
print(f"x(0) = {float(x0):.12f}   (want 1)")
print(f"x'(0) = {float(v0):.12f}   (want 0)")
print("both exact, before any training")
"""),
            md("""
## Step 3 · Collocation points and the residual

The physics loss is evaluated at **collocation points**: times we choose,
where we ask whether the equation holds. They are not data — nothing is known
about the solution there. They are just where we check.

$$\\mathcal{L} = \\frac{1}{N}\\sum_i \\left[\\ddot x_\\theta(t_i) + \\omega^2 x_\\theta(t_i)\\right]^2$$
"""),
            code("""
OMEGA = 2.0

def residual(t, f=None):
    t = t.requires_grad_(True)
    x = x_hard(t, f)
    dx = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    d2x = torch.autograd.grad(dx.sum(), t, create_graph=True)[0]
    return d2x + OMEGA**2 * x

t_colloc = torch.linspace(0, 3, 200)
print("collocation points:", len(t_colloc))
print("residual before training (should be large):",
      float(torch.mean(residual(t_colloc.clone())**2)))
"""),
            md("## Step 4 · Train, and watch it obey the equation"),
            code("""
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
history = []
for epoch in range(3000):
    opt.zero_grad()
    loss = torch.mean(residual(t_colloc.clone())**2)
    loss.backward(); opt.step(); sched.step()
    if epoch % 100 == 0:
        history.append((epoch, float(loss)))
print(f"final residual loss: {float(loss):.3e}")
"""),
            code("""
grid = torch.linspace(0, 3, 400)
with torch.no_grad():
    learned = x_hard(grid).numpy()
truth = np.cos(OMEGA * grid.numpy())

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
axes[0].plot(grid, truth, lw=3, color=P.INK_MUTED, label="exact  cos(wt)")
axes[0].plot(grid, learned, color=P.ARM_COLOR["pinn"], label="PINN")
axes[0].set_xlabel("t"); axes[0].set_ylabel("x(t)"); axes[0].legend()
axes[0].set_title("The network learned cos(wt) from the equation alone")
e, l = zip(*history)
axes[1].semilogy(e, l, color=P.ARM_COLOR["pinn"])
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("residual loss")
axes[1].set_title("Physics loss")
plt.show()
print(f"max |error| over the interval: {np.max(np.abs(learned - truth)):.2e}")
"""),
            md("""
## What just happened

The network never saw a single value of $\\cos$. It was handed an equation,
some points at which to satisfy it, and initial conditions it could not
violate — and it produced the solution.

**This is the forward problem**: the law is fully known and we want the
solution. It is also the case where a PINN is *least* competitive: a standard
ODE integrator solves this in microseconds to machine precision, while the
network took thousands of epochs to reach ~1e-3.

So why use one? Because the same machinery does something integrators cannot,
which is the subject of [T2](T2_forward_and_inverse.ipynb): **recovering
unknown constants in the equation from sparse, noisy measurements**.
"""),
        ],
        "T1 - what a PINN is",
    )


# ---------------------------------------------------------------------------
# T2 -- forward and inverse
# ---------------------------------------------------------------------------


def t2_forward_and_inverse():
    return _nb(
        [
            md("""
# T2 · The two jobs: solving an equation, and recovering a constant

[T1](T1_what_is_a_pinn.ipynb) solved a fully known equation — the **forward**
problem, where an ODE integrator beats a PINN easily.

The **inverse** problem is the one worth caring about. The law's *form* is
known; one of its constants is not, and you have a handful of noisy
measurements. Now the network has two jobs at once:

1. satisfy the equation, and
2. pass near the data,

and the unknown constant is trained alongside the weights.
"""),
            code(SETUP),
            md("""
## The setup: eight noisy points, and an unknown frequency

The truth is $\\omega = 2$ and we pretend not to know it. Eight measurements,
5% noise. A curve fit would recover $\\omega$ from these too — the point here
is the *mechanism*, which generalises to equations that have no closed-form
solution to fit.
"""),
            code("""
OMEGA_TRUE = 2.0
rng = np.random.default_rng(0)
t_data = np.sort(rng.uniform(0, 3, 8))
x_data = np.cos(OMEGA_TRUE * t_data) + rng.normal(0, 0.05, 8)
t_d = torch.tensor(t_data); x_d = torch.tensor(x_data)
print("t:", t_data.round(3)); print("x:", x_data.round(3))
"""),
            md("""
## The unknown constant is just another parameter

`omega` is a `torch.nn.Parameter`, optimised by the same Adam step as the
weights. It is trained **in log space**: it is positive by physics, and the
log keeps it positive without a constraint while making the search
scale-free.
"""),
            code("""
def mlp(width=32, depth=3):
    layers = [torch.nn.Linear(1, width), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))

torch.manual_seed(0)
net = mlp()
log_omega = torch.nn.Parameter(torch.tensor(0.0))   # omega = exp(0) = 1, wrong on purpose

def omega():
    return torch.exp(log_omega)

def x_of(t):
    return 1.0 + t**2 * net(t.unsqueeze(-1)).squeeze(-1)

print(f"omega starts at {float(omega()):.3f}, truth is {OMEGA_TRUE}")
"""),
            md("""
## Two terms in the loss, and the weight between them

$$\\mathcal{L} = \\underbrace{\\frac{1}{N_d}\\sum (x_\\theta(t_i) - x_i)^2}_{\\text{data}}
 \\;+\\; w_{\\text{phys}} \\underbrace{\\frac{1}{N_c}\\sum \\left[\\ddot x_\\theta + \\omega^2 x_\\theta\\right]^2}_{\\text{physics}}$$

`w_phys` is the dial this whole repository is built around. Turn it to zero
and you have an ordinary curve fit that happens to carry an unused symbol
called `omega`; turn it up and the equation dominates.
"""),
            code("""
t_colloc = torch.linspace(0, 3, 200)

def train(w_phys, epochs=4000, seed=0):
    torch.manual_seed(seed)
    global net, log_omega
    net = mlp(); log_omega = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(net.parameters()) + [log_omega], lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    track = []
    for epoch in range(epochs):
        opt.zero_grad()
        data = torch.mean((x_of(t_d) - x_d)**2)
        t = t_colloc.clone().requires_grad_(True)
        x = x_of(t)
        dx = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
        d2x = torch.autograd.grad(dx.sum(), t, create_graph=True)[0]
        phys = torch.mean((d2x + omega()**2 * x)**2)
        (data + w_phys * phys).backward(); opt.step(); sched.step()
        if epoch % 50 == 0:
            track.append((epoch, float(omega())))
    return float(omega()), track, float(data)

omega_hat, track, data_loss = train(w_phys=1.0)
print(f"recovered omega = {omega_hat:.4f}   truth {OMEGA_TRUE}   "
      f"error {abs(omega_hat-OMEGA_TRUE)/OMEGA_TRUE*100:.2f}%")
"""),
            md("""
## The dial, swept

This is the single most important plot in the subject. Watch the recovered
constant as `w_phys` goes from 0 to large.
"""),
            code("""
rows = []
for w in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
    om, _, dl = train(w_phys=w)
    rows.append((w, om, abs(om - OMEGA_TRUE) / OMEGA_TRUE * 100, dl))
    print(f"w_phys={w:<7g} omega={om:7.4f}  error={rows[-1][2]:7.2f}%  data loss={dl:.2e}")
"""),
            code("""
w, om, err, dl = zip(*rows)
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.axhline(OMEGA_TRUE, color=P.INK_2, lw=1.4)
ax.annotate("true omega", xy=(1.2e-3, OMEGA_TRUE), xytext=(0, 6),
            textcoords="offset points", color=P.INK_2, fontsize=8.5)
ax.plot([max(x, 1e-4) for x in w], om, "-o", color=P.ARM_COLOR["pinn"])
ax.set_xscale("log"); ax.set_xlabel("w_phys  (0 shown at 1e-4)")
ax.set_ylabel("recovered omega")
ax.set_title("With no physics term the constant means nothing")
plt.show()
"""),
            md("""
## The finding

At `w_phys = 0` the network fits the eight points perfectly well and
`omega` drifts to whatever is left over — **the data loss barely changes
while the recovered constant is badly wrong**. The physics term is not a
regulariser you tune for accuracy. It is what makes the parameter
*identifiable* at all.

This repository measures the same effect on real data: at `w_phys = 0` the
recovered `GM_sun` is 19.5% wrong and the CMB temperature 1.33% wrong, while
held-out error hardly moves. A PINN that fits well is not thereby measuring
anything.

Next: [T3](T3_gravity.ipynb) puts this on a real ephemeris.
"""),
        ],
        "T2 - forward and inverse",
    )


# ---------------------------------------------------------------------------
# T3 -- gravity: the law is exact
# ---------------------------------------------------------------------------


def t3_gravity():
    return _nb(
        [
            md("""
# T3 · Gravity — where the law is exact

The first of three physics domains, in rising order of how badly the law can
let you down. Here it cannot: Kepler's third law

$$P = 2\\pi\\sqrt{a^3 / GM}$$

is exact for a two-body system, and the only unknown is `GM`. This is the
**best case** for a physics prior, and it is worth seeing what that best case
actually buys — which is less than you might expect.
"""),
            code(
                SETUP
                + """
from physprior.problems.gravity import kepler
from physprior.benchmark.protocol import fit_arm, split_random
from physprior.io import load_table
prob, meta = kepler.problem()
print(f"track: {prob.track},  {len(prob)} planets,  recovering {[p.name for p in prob.params]}")
print(f"published GM_sun = {meta['published_GM_sun']:.6e} m^3/s^2")
"""
            ),
            md("""
## First, the control: simulations where the answer is exact

Every real-data number in this repository is read against a **simulation**
where the law is exactly what was put in. When a method fails there, the
failure is the method's own -- there is no other candidate.
"""),
            code("""
from IPython.display import HTML, Image, display
display(HTML('<table><tr>'
  '<td><img src="../../figures/gravity/three_body_figure8.gif" width="330"></td>'
  '<td><img src="../../figures/gravity/three_body_chaotic.gif" width="330"></td>'
  '</tr><tr>'
  '<td align="center"><sub>the figure-eight choreography</sub></td>'
  '<td align="center"><sub>the same, perturbed by 1 part in 10^9</sub></td>'
  '</tr></table>'))
"""),
            md("""
Symbolic regression recovers the force-law exponent as **-1.9999969** and
`GM` to **0.6 ppb** from a simulated orbit. That is the floor. So when the
same machinery returns `GM_sun` from the real ephemeris with a tens-of-ppm
residual, the residual is not the method -- it is the two-body formula
neglecting the planets' masses.

## Five arms on the same data

Every track in this repository is run through the same five arms, so that
"physics-informed" is compared against something rather than asserted:

| arm | knows the law? | returns a constant? |
|---|---|---|
| `oracle` | published law **and** published constant — the ceiling, not a competitor | — |
| `physics` | the law, constants fitted by `curve_fit` | yes, with a covariance |
| `pinn` | `y = law(x; θ) + σ_y·NN(x)` — law plus a learned correction | yes |
| `sr` | **no** — symbolic regression searches for a law | sometimes |
| `nn` | no — a tuned MLP | no |
"""),
            code("""
idx = np.arange(len(prob))
fits = {a: fit_arm(a, prob, idx, seed=11) for a in ("oracle", "physics", "pinn", "nn")}

published = meta["published_GM_sun"]
named = [(a, f) for a, f in fits.items() if f.params.get("GM") is not None]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.axvline(0, color=P.INK_2, lw=1.4)
ax.annotate("published GM", xy=(0, len(named) - 0.45), xytext=(6, 0),
            textcoords="offset points", fontsize=8.4, color=P.INK_2, va="center")
for i, (arm, f) in enumerate(named):
    y = len(named) - 1 - i
    ppm = (f.params["GM"] - published) / published * 1e6
    ax.scatter([ppm], [y], s=90, color=P.ARM_COLOR.get(arm, P.INK_MUTED),
               edgecolor=P.SURFACE, linewidth=1.5, zorder=3)
    ax.annotate(f"  {ppm:+.1f} ppm", xy=(ppm, y), va="center", fontsize=8.5,
                color=P.ARM_COLOR.get(arm, P.INK_MUTED))
ax.set_yticks(range(len(named)), [a for a, _ in reversed(named)])
ax.set_xlabel("deviation from the published GM_sun  (ppm)")
ax.grid(axis="y", visible=False); ax.set_xlim(-90, 90)
ax.set_ylim(-0.6, len(named) - 0.15)   # headroom, or the note clips
ax.set_title("Only three of the five arms return a constant at all")
plt.show()

for arm, f in fits.items():
    print(f"{arm:8s} {f.expression or 'no closed form -- nothing to argue with'}")
"""),
            md("""
## The PINN here is a *correction*, not a solver

Note the shape of the `pinn` arm. It is not solving a differential equation
as in T1 — it is

$$y = \\text{law}(x;\\theta) + \\sigma_y\\,\\mathrm{NN}(x)$$

with the physical constant $\\theta$ trainable, and a loss that penalises the
size of the correction:

$$\\mathcal{L} = \\frac{\\mathrm{MSE}}{\\sigma_y^2} + w_{\\text{phys}}\\,\\overline{\\mathrm{NN}^2}$$

That makes `w_phys` a **continuous dial between two named methods**: at
$w_{\\text{phys}}\\to\\infty$ the correction is crushed and the arm *is* the
classical fit; at $w_{\\text{phys}}=0$ it is a black box wearing a physics hat.

Both forms are PINNs. Which one you want depends on whether your law is an
algebraic relation (this) or a differential equation (T1, and T4).
"""),
            code("""
sweep = load_table("gravity/kepler", "sweep_physics_weight")
g = sweep.groupby("w_phys")[["nrmse_out", "err_GM_pct"]].median()
print(g.to_string(float_format=lambda v: f"{v:.5g}"))
"""),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
axes[0].semilogx([max(w, 1e-4) for w in g.index], g.err_GM_pct.abs(), "-o",
                 color=P.ARM_COLOR["pinn"])
axes[0].set_yscale("log"); axes[0].set_xlabel("w_phys (0 shown at 1e-4)")
axes[0].set_ylabel("|error| in recovered GM  (%)")
axes[0].set_title("The constant")
axes[1].semilogx([max(w, 1e-4) for w in g.index], g.nrmse_out, "-o",
                 color=P.ARM_COLOR["physics"])
axes[1].set_yscale("log"); axes[1].set_xlabel("w_phys (0 shown at 1e-4)")
axes[1].set_ylabel("held-out nRMSE"); axes[1].set_title("The fit")
plt.show()
"""),
            md("""
## Watching it happen

Training is recorded, not just reported. The left panel is the fit; the right
is the physical constant walking toward its published value as the physics
term pulls on it.
"""),
            code("""
# <img>, not Image(): the browser resolves it relative to THIS notebook,
# and the GIF stays out of the .ipynb instead of being embedded as base64.
display(HTML('<img src="../../figures/gravity/pinn_learning_orbit.gif" width="620">'))
"""),
            md("""
## The finding, on real data

The left panel falls off a cliff between `w_phys = 0` and `0.01`; the right
panel barely moves. **The held-out error cannot tell you whether your
recovered constant is any good.**

That is the same lesson as T2, now on eight real planets from the JPL DE441
ephemeris rather than a toy.

### What the physics prior buys here

With the law exact and the data clean, remarkably little in-distribution —
a tuned black box is competitive inside the training range. The prior earns
its place *outside* it, and in returning a number a physicist can argue with.

See [`docs/gravity/`](../../docs/gravity) for the track in full. Next:
[T4](T4_relativity.ipynb), where the law stops being exact.
"""),
        ],
        "T3 - gravity",
    )


# ---------------------------------------------------------------------------
# T4 -- relativity: the law is a truncated expansion
# ---------------------------------------------------------------------------


def t4_relativity():
    return _nb(
        [
            md("""
# T4 · Relativity — where the law is an approximation

Gravity's law was exact. This one is a **series**, and you get to choose where
to stop:

$$\\dot f = \\frac{96}{5}\\pi^{8/3}\\left(\\frac{G\\mathcal{M}}{c^3}\\right)^{5/3} f^{11/3}
\\left[1 + \\text{(1PN)} + \\text{(1.5PN tail)} + \\text{(2PN)} + \\cdots\\right]$$

Truncating early does not give you a wrong-looking fit. It gives you a
**confident wrong answer**, which is far more dangerous.
"""),
            code(
                SETUP
                + """
from physprior.io import load_json
meta = load_json("relativity/gw150914", "meta")
print(f"GW150914: {meta['n_cycles']} usable cycles, SNR {meta['snr']}")
print(f"published chirp mass (detector frame): {meta['published_Mc_detector']:.3f} Msun")
"""
            ),
            md("""
## The control: a simulation where GR is exactly what was put in

`d2u/dphi2 + u = GM/h^2 + (3GM/c^2)u^2`, integrated with the GR term on and
off. The difference between the two runs is the precession, and it comes out
at 42.98 arcsec/century -- the measured value.
"""),
            code("""
from IPython.display import HTML, display
display(HTML('<img src="../../figures/relativity/schwarzschild_precession.gif" width="560">'))
"""),
            md("""
Locating the perihelion here is **root-finding, not parabola-fitting**. The
shift is 5e-7 rad per orbit and fitting a parabola to a sampled grid is good
to ~1e-6 -- bigger than the effect. The first version of this simulation duly
reported a Newtonian "precession" 20% larger than the GR one.

## The post-Newtonian ablation

The same data, the same code, the same fitting — only the order at which the
expansion is truncated changes.
"""),
            code("""
import pandas as pd
abl = pd.DataFrame(meta["pn_ablation"])
print(abl[["label", "Mc", "Mc_sigma", "bias_Msun", "bias_sigma", "rmse_hz", "converged"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
"""),
            code("""
# The 1PN row is EXCLUDED because its fit pinned to a bound: a parameter at
# its bound has not converged, whatever the optimiser reports, and plotting
# it as a point would imply a measurement that was never made.
ok = abl[abl.converged]
x = np.arange(len(ok))
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
axes[0].axhline(meta["published_Mc_detector"], color=P.INK_2, lw=1.4)
axes[0].annotate("published", xy=(x[0], meta["published_Mc_detector"]),
                 xytext=(0, 6), textcoords="offset points",
                 color=P.INK_2, fontsize=8.5)
axes[0].errorbar(x, ok.Mc, yerr=ok.Mc_sigma, fmt="o",
                 color=P.ARM_COLOR["physics"], capsize=3, markersize=8,
                 markeredgecolor=P.SURFACE, markeredgewidth=1.2)
axes[0].set_xticks(x, ok.label, rotation=12)
axes[0].set_ylabel("recovered chirp mass (Msun)")
axes[0].set_title("The answer moves by 9 solar masses")
axes[1].plot(x, ok.rmse_hz, "-o", color=P.ARM_COLOR["pinn"])
axes[1].set_xticks(x, ok.label, rotation=12)
axes[1].set_ylabel("RMSE of the fit (Hz)")
axes[1].set_title("The fit quality barely notices")
plt.show()
print(f"chirp mass spread across orders: {ok.Mc.max() - ok.Mc.min():.2f} Msun")
print(f"RMSE spread across orders:       {ok.rmse_hz.max() - ok.rmse_hz.min():.3f} Hz")
"""),
            md("""
## The finding: goodness of fit does not diagnose a wrong law

Note the row that is **missing from the plot**. The 1PN fit pinned its
parameter to a bound at `Mc = 200` with a formal error of 710 — the optimiser
reported a result, and it is not one. This project's rule is that a parameter
at its bound has not converged whatever the optimiser says, so it is marked
and excluded rather than drawn as a point.

At Newtonian order the recovered chirp mass is biased by about **+9 M☉** with
a formal error of 4.1 — the fit reports high confidence in a wrong number.
Adding the 1.5PN tail and 2PN terms removes the bias **while the RMSE hardly
moves**.

If you are ever tempted to choose a physical model by held-out error, this is
the counter-example. The residual is dominated by measurement noise, not by
the missing physics, so it cannot see the missing physics.

## This track's PINN is a different animal

Because the law here *is* a differential equation, `relativity/gw150914` uses
the **residual PINN of Raissi et al. (2019)** — the T1 form — rather than the
law-plus-correction form of T3:

```python
y  = net(t_collocation)
dy = torch.autograd.grad(y, t_collocation, create_graph=True)[0]
residual = dy - fdot_pn(y, Mc)          # Mc is an nn.Parameter
loss = data_mse + w_phys * (residual ** 2).mean()
```

`create_graph=True` is what lets the residual itself be differentiated during
back-propagation, so `Mc` receives a gradient *through the physics term*.
That single flag is the difference between training the constant and not.

## And a warning that has nothing to do with networks

The chirp mass is extracted through a chain of signal-processing choices. One
— the envelope SNR a cycle must clear — was set to 2.0 a priori and looked
perfectly reasonable. Injecting a **known** chirp mass into the **real
detector noise** and running the identical code:
"""),
            code("""
from physprior.io import load_table
cal = load_table("relativity", "threshold_calibration")
print(cal.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
"""),
            md("""
At a threshold of 2.0 the pipeline recovers 9.3 M☉ for an injected 31.2 — a
**−70% bias**. The cut was re-chosen on injections, never on the real event.

> A pipeline you have not injected into has no error budget, and a constant it
> returns is a number with no uncertainty attached to the method that produced
> it.

See [`docs/relativity/`](../../docs/relativity). Next:
[T5](T5_quantum_wavefunction.ipynb), where there is no data at all.
"""),
        ],
        "T4 - relativity",
    )


# ---------------------------------------------------------------------------
# T5 -- quantum: learning a wave function
# ---------------------------------------------------------------------------


def t5_quantum_wavefunction():
    return _nb(
        [
            md("""
# T5 · Quantum — learning a wave function, with no data at all

The hardest of the three, and the purest. T3 had data and a law. T4 had data
and a *truncated* law. Here there is **no data on the right-hand side at
all** — only an operator equation and a boundary:

$$-\\tfrac{1}{2}\\psi''(x) + V(x)\\,\\psi(x) = E\\,\\psi(x), \\qquad \\psi(a)=\\psi(b)=0$$

and **two** unknowns to find together: the wave function $\\psi$ *and* its
eigenvalue $E$. This is an eigenvalue problem, not an initial-value one, and
it breaks in four distinct ways.
"""),
            code(
                SETUP
                + """
from physprior.methods.eigen_pinn import fit_eigen_pinn, fit_spectrum, WaveFunction
from physprior.problems.quantum import schrodinger as S
"""
            ),
            md("""
## The system, moving

Before the eigenvalue problem, the same equation run forward in time: a wave
packet meeting a barrier, by split-operator FFT, unitary to 5e-15.
"""),
            code("""
from IPython.display import HTML, display
display(HTML('<img src="../../figures/quantum/tunnelling.gif" width="560">'))
"""),
            md("""
## Trap 1 · ψ = 0 solves the equation exactly

Put the zero function into the residual and it vanishes identically. A naive
`mean(residual²)` loss therefore has a perfect, useless global minimum, and
that is where it will go.

The fix is to make the objective a **ratio of inner products**, so scale
cancels and the zero function is not in the domain:

$$E[\\psi] = \\frac{\\langle\\psi|\\hat H|\\psi\\rangle}{\\langle\\psi|\\psi\\rangle}
\\qquad\\text{(the Rayleigh quotient)}$$

Minimising this *is* the variational principle: its minimum is the ground
state energy. `method="rayleigh"` does this; `method="residual"` keeps `E` as
a trainable parameter and divides by the norm for the same reason.
"""),
            md("""
## Trap 2 · The boundary condition, hard rather than penalised

Same principle as T1's initial conditions, and the same payoff:

$$\\psi_\\theta(x) = (x-a)(b-x)\\,\\mathrm{NN}(x)$$

vanishes at both walls for *any* network, exactly, at every step.
"""),
            code("""
net = WaveFunction(0.0, 1.0)
ends = torch.tensor([0.0, 1.0])
print("psi at the walls, untrained:", net(ends).detach().numpy())
print("exact, and no boundary weight to tune")
"""),
            md("## Now solve it — the infinite square well, where $E_n = n^2\\pi^2/2$"),
            code("""
state = fit_eigen_pinn(lambda x: np.zeros_like(x), 0.0, 1.0, epochs=2500, seed=11)
exact = np.pi**2 / 2
print(f"E learned = {state.energy:.5f}")
print(f"E exact   = {exact:.5f}")
print(f"error     = {abs(state.energy-exact)/exact*100:.4f}%")
"""),
            code("""
psi = state.normalised()
if psi[len(psi)//2] < 0: psi = -psi
truth = np.sqrt(2) * np.sin(np.pi * state.x)

fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(state.x, truth, lw=3, color=P.INK_MUTED, label="exact  sqrt(2) sin(pi x)")
ax.plot(state.x, psi, color=P.ARM_COLOR["pinn"], label="PINN")
ax.set_xlabel("x"); ax.set_ylabel("psi(x)"); ax.legend()
ax.set_title("A wave function learned from an operator equation")
plt.show()
print(f"max |error| in psi: {np.max(np.abs(psi - truth)):.3e}")
"""),
            md("""
## Trap 3 · Excited states need orthogonality — and the weight is not free

Both objectives are minimised by the **ground** state, so asking for level 2
gets you level 1 again. The fix is a penalty on overlap with what you already
found:

$$\\mathcal{L} \\mathrel{+}= w\\sum_j \\frac{\\langle\\psi|\\psi_j\\rangle^2}{\\langle\\psi|\\psi\\rangle}$$

But $w$ is **not** a knob to twiddle. Compare the two options available to the
optimiser:

* stay on the ground state: cost $E_0 + w\\cdot 1$
* climb to the next level: cost $E_1 + 0$

so any $w < E_1 - E_0$ makes collapse the *cheaper* option. On this well the
first gap is 14.8, and $w=10$ duly returned level 2 as an exact copy of level
1, overlap 1.0000. The implementation scales $w$ by the energies already
found for exactly this reason.
"""),
            code("""
states = fit_spectrum(lambda x: np.zeros_like(x), 0.0, 1.0, n_levels=4,
                      epochs=2500, seed=11)
print(f"{'n':>2} {'E PINN':>10} {'E exact':>10} {'err %':>7} {'overlap':>8} {'conv':>6} {'tries':>6}")
for k, st in enumerate(states, start=1):
    e = k**2 * np.pi**2 / 2
    print(f"{k:>2} {st.energy:10.5f} {e:10.5f} {abs(st.energy-e)/e*100:7.3f} "
          f"{st.max_overlap:8.4f} {str(st.converged):>6} {st.attempts:>6}")
"""),
            code("""
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for k, st in enumerate(states, start=1):
    p = st.normalised()
    if np.trapezoid(p * np.sqrt(2)*np.sin(k*np.pi*st.x), st.x) < 0: p = -p
    ax.plot(st.x, p + 3*(k-1), color=P.ARM_COLOR["pinn"])
    ax.plot(st.x, np.sqrt(2)*np.sin(k*np.pi*st.x) + 3*(k-1), lw=3, alpha=0.35,
            color=P.INK_MUTED, zorder=1)
    ax.annotate(f"n = {k}", xy=(1.01, 3*(k-1)), fontsize=8.5, color=P.INK_2)
ax.set_xlabel("x"); ax.set_yticks([]); ax.set_xlim(0, 1.12)
ax.set_title("Four eigenstates, each orthogonal to the ones below it")
ax.plot([], [], lw=3, alpha=0.35, color=P.INK_MUTED, label="exact")
ax.plot([], [], color=P.ARM_COLOR["pinn"], label="PINN")
ax.legend(loc="lower right")
plt.show()
"""),
            md("""
## Trap 4 · Spectral bias — the best-known PINN failure mode

Look at the `tries` column. The fourth level often needs a restart, and
sometimes comes back as a copy of the ground state *even though the objective
still prefers the true answer* — falling back costs the orthogonality penalty,
by then around 459, against the true level's 79.

So it is not the objective that failed. It is the **optimiser**: reshaping one
lobe into four means crossing a barrier in function space, and networks learn
low frequencies long before high ones. That is spectral bias, and it is why
Fourier features exist.

The collapse is detected, not hidden: `fit_spectrum` measures the overlap,
retries from another seed, and returns a still-collapsed level with
`converged = False`.

## Trap 5 · You cannot validate a PINN from inside the PINN

`converged` above is an **overlap** check: it catches a level that came back
as a copy of a lower one. It does not catch a level that is simply *wrong* —
orthogonal to everything below it, and still not an eigenstate. That happens,
and on some seeds level 4 returns ~95 against a true 79, a 20% error, with an
overlap of 0.03 and a `converged` flag of `True`.

The obvious remedy is the residual, `||H psi - E psi|| / ||psi||`.
**It does not work.** Measured across two seeds where
every level is good to better than 0.25%:

| level | error | residual / E |
|---|---|---|
| 1 | 0.000 % | 0.004 |
| 2 | 0.041 % | 0.091 |
| 3 | 0.060 % | 0.131 |
| 4 | 0.045 % | 0.117 |

The residual of a *correct* state spans 0.004 to 0.21 here, because at this
training budget it is dominated by the network's finite capacity rather than
by wrongness. A wrong answer can easily have a smaller residual than a right
one.

> **The internal diagnostics of a PINN cannot tell you whether it is right.**
> Low loss, small residual and a satisfied constraint are all compatible with
> a badly wrong answer.

This is why every track in this repository is checked against something
independent — a closed form, a second method, or an injection with a known
answer. It is also the answer to "how do you know your PINN worked?":
*you do not, from the inside*.

## The reality check

`solve_1d` diagonalises a tridiagonal matrix and gets these same eigenvalues
in **milliseconds**, more accurately:
"""),
            code("""
import time
t0 = time.time(); ref = S.infinite_well(length=1.0, n_levels=4); dt = time.time() - t0
print(f"diagonalisation: {dt*1000:.1f} ms, max relative error {ref.max_rel_error:.2e}")
print(f"PINN:            {sum(s.seconds for s in states):.0f} s, "
      f"max relative error {max(abs(s.energy - e)/e for s, e in zip(states, ref.energy)):.2e}")
"""),
            md("""
**The PINN is thousands of times slower and less accurate.** On a 1-D problem
with a tridiagonal matrix, it should never be your choice.

So why learn it? Because the method does not care about dimension or mesh.
The same twenty lines extend to geometries where no such matrix exists, to
potentials known only pointwise, and — the case that actually matters — to the
**inverse** problem: given a measured spectrum, recover $V(x)$. That is the
same trick as T2's `omega`, applied to a whole function.

And because knowing *where it breaks* is most of the expertise. Everything in
this notebook that went wrong — the trivial solution, the penalty weight below
the gap, the collapse at level 4 — is a general property of PINNs, shown here
in a problem whose answer is known to nine digits.

Next: [T6](T6_when_pinns_fail.ipynb) collects the failures this repository has
measured on real data.
"""),
            md("""
---

## The inverse problem: recover the potential from the spectrum

This is the one worth caring about. The forward problem has a better solver;
the inverse has none. Hand it only the energies — never the functional form,
never the word "harmonic" — and ask for `V(x)`.
"""),
            code("""
from physprior.methods.eigen_pinn import fit_inverse_potential
energies = np.arange(6) + 0.5          # all the method is told
inv = fit_inverse_potential(energies, -6.0, 6.0, n_collocation=256,
                            epochs=3000, seed=11)
print("target  :", np.round(inv.energies_target, 4))
print("achieved:", np.round(inv.energies_achieved, 4), " (by diagonalising V_hat)")
print(f"spectrum error: {inv.spectrum_error*100:.2f}%")
for x0, v_true in ((0.0, 0.0), (1.0, 0.5), (2.0, 2.0), (3.0, 4.5)):
    print(f"  V({x0}) = {np.interp(x0, inv.x, inv.v):7.3f}   true {v_true}")
"""),
            code("""
fig = P.fig_inverse_potential(inv, truth=lambda x: 0.5*x**2,
                              title="Recovered: V(x) = x^2/2")
plt.show()
"""),
            md("""
### Three things this problem teaches that the forward one cannot

**1 · One spectrum does not determine a potential.** Borg-Marchenko: two
spectra are needed in general, and "can one hear the shape of a drum?" is the
same question. Here symmetry is imposed *architecturally* — `V` is evaluated
at `|x - centre|` — which buys **identifiability**, not accuracy. Without it
the problem is genuinely ill-posed and no amount of training fixes that.

**2 · Where no state lives, the data is silent.** Every term of the residual
is proportional to psi, so out in the classically forbidden tails the
spectrum says nothing about V at all. An unconstrained network puts a bump
there, and that bump manufactures spurious bound states that corrupt the very
spectrum you were matching. The first run of this produced a potential that
turned *over* at large $|x|$ and returned a degenerate pair where the truth
has none.

The fix is a Tikhonov term on the curvature of V: it adds no information, it states a
*preference* for the smoothest potential consistent with the data. That is a
prior and it must be reported as one.

**3 · When a differentiable forward model exists, use it instead.**

| | residual PINN | differentiable eigensolver |
|---|---|---|
| spectrum error | 0.17 | **0.0034** |
| time | 100 s | **18 s** |

The PINN has to represent every state with its own network and satisfies the
eigenvalue equation only approximately, so the eigenvalues it reports are not
the ones its potential actually has. Differentiating through
`torch.linalg.eigvalsh` leaves nothing to approximate on the forward side and
puts every bit of the optimisation into `V`.

**That is the scope of a PINN.** It is not a better eigensolver. It is
what remains when there is no eigensolver — an unmeshable geometry, a forward
model you cannot differentiate, a physical law known only as a residual. Both
methods are kept in this repository so that claim stays a measurement rather
than an opinion.
"""),
        ],
        "T5 - quantum wave function",
    )


# ---------------------------------------------------------------------------
# T6 -- the failure catalogue
# ---------------------------------------------------------------------------


def t6_when_pinns_fail():
    return _nb(
        [
            md("""
# T6 · When PINNs fail — a catalogue, measured

The previous five notebooks built PINNs up. This one is the part an interview
actually probes: **when does this not work, and how would you know?**

Every number here was measured in this repository, on real data, with the
seed discipline stated: tune on 3/7/19, report on 11/23/42, never select on a
reported seed.
"""),
            code(
                SETUP
                + """
import pandas as pd
from physprior.io import load_table
from physprior.reporting import conclusions as C
"""
            ),
            md("""
## 0 · The two shapes a PINN comes in

Before the failures, the anatomy. These are not variants of one architecture
— they answer different questions, and `w_phys` means something different in
each.
"""),
            code("""
P.fig_pinn_anatomy(); plt.show()
"""),
            md("""
Panel **A** is T1, T4 and T5: the law is a differential equation, the network
*is* the solution, and the physical constant sits inside the residual so it
receives a gradient through the physics term. Panel **B** is T3: the law is an
algebraic relation, the network is a *correction* to it, and `w_phys` is a
continuous dial between the classical fit and a black box.

## 1 · The network eats the physics if you let it

Already seen in T2 and T3 — repeated because it is the one that produces
*publishable-looking* wrong answers.

| problem | constant | error at `w_phys = 0` | error at `w_phys ≥ 0.01` |
|---|---|---|---|
| gravity | `GM_sun` | **19.5 %** | 0.005 % |
| quantum | `T_CMB` | **1.33 %** | 0.017 % |

Held-out error barely moves across that range. **A PINN that fits well is not
thereby measuring anything.**

## 2 · Fit quality cannot diagnose a wrong law

T4's post-Newtonian ablation: the chirp mass moves by 9 M☉ across truncation
orders while the RMSE hardly notices. If your model-selection criterion is
held-out error, you will select the wrong physics.

## 3 · A result still moving with step size is not a result
"""),
            code("""
gr = load_table("relativity/mercury", "gr_convergence")
P.fig_alpha_convergence(gr.to_dict("records")); plt.show()
"""),
            md("""
At a 3-hour step with a 4th-order stencil, the GR coefficient comes out at
**α = 1.1343 ± 0.0024** — a 13.4% violation of general relativity at **56
formal sigma**. It is entirely finite-difference truncation error. With a
6th-order stencil α agrees with Einstein to one part in 10⁴.

The error bar was never wrong. It was correctly answering a question about
the *statistics* of a model that was *systematically* wrong.

> The result is not α; it is α once it has stopped moving.

## 4 · Which switches actually help — measured, not assumed

Seven common PINN improvements were ablated one at a time on the tuning
seeds. The pre-registered ranking got the top two backwards.
"""),
            code("""
import glob
from physprior.benchmark.ablation import summarise
abl = pd.concat([pd.read_csv(f) for f in glob.glob("results/*/*/tune/ablation.csv")])
P.fig_ablation_effect(summarise(abl)); plt.show()
"""),
            md("""
Every dot is one (track, protocol question). Left of the line the switch
helped, right of it it hurt, and the distance is the effect size on a log
scale. Two clusters entirely on one side, two entirely on the other, and one
sitting exactly on the line.

And what the shipped switch bought, per track, on the reporting seeds after
re-running the whole pipeline:
"""),
            code("""
from IPython.display import HTML, display
display(HTML('<img src="../../figures/phase2_improvement.png" width="800">'))
"""),
            md("""
| option | verdict |
|---|---|
| **gradient-norm loss balancing** (Wang et al. 2021) | **ships** — helps 5 cells, hurts none, up to 101× |
| deep ensembles | helps, but 5× the compute for a further 1.3–1.4× |
| **early stopping** | **rejected** — hurt 4 cells, helped none |
| **Fourier features** | **rejected** — made one track **98× worse** |
| **L-BFGS refinement** | **rejected** — its proposal never lowered the loss |

Two things worth saying out loud in an interview:

* Fourier features are the standard remedy for the spectral bias of T5 — and
  on *these* tracks, whose targets are smooth, they were catastrophic. A
  remedy is only a remedy for the disease it treats.
* The pre-registered ranking in `docs/plans/PLAN.md` put early stopping first
  and balancing fourth. The ablation reversed them. The ranking was left in
  the document rather than quietly reordered.

## 5 · Where the prior pays, and where it does not
"""),
            code("""
frame = C.verdict_frame("all")
counts = frame.verdict.str.split(" ").str[0].value_counts()
fig, ax = plt.subplots(figsize=(6.2, 3.2))
colours = {"WIN": P.ARM_COLOR["pinn"], "TIE": P.INK_MUTED}
ax.barh(list(counts.index), list(counts.values),
        color=[colours.get(k, P.INK_MUTED) for k in counts.index],
        edgecolor=P.SURFACE, linewidth=2, height=0.55)
for k, v in counts.items():
    ax.annotate(f"  {v}", xy=(v, k), va="center", fontsize=9, color=P.INK_2)
ax.set_xlabel("protocol questions"); ax.grid(axis="y", visible=False)
ax.set_title("Most questions do not separate the arms at all")
plt.show()
print(f"{counts.get('TIE', 0)} of {len(frame)} questions sit inside the seed spread")
"""),
            md("""
Read the `verdict` column carefully. Most questions are **ties** — the gap
between arms is smaller than the seed-to-seed spread, and calling those wins
would be noise-mining.

The pattern across this repository, stated plainly:

1. **Inside the training range, with enough clean data, a tuned black box is
   competitive.** A physics prior buys little there. Claims to the contrary
   usually compare against an untuned baseline.
2. **Outside it the gap is orders of magnitude — but the mechanism is
   identifiability, not the mere presence of a law.** On `quantum/cmb`,
   out-of-band error falls ~56× as the fitted band reaches the Rayleigh-Jeans
   regime *while in-band error gets worse*. On `relativity/gw150914` —
   the counter-control — extrapolation from four faint cycles defeats every
   fitted arm, physics included.
3. **Only the physics arms return something a physicist can argue with**, and
   three times here the argument was worth having: QED in hydrogen, the PN
   expansion in GW150914, and a truncation error masquerading as a refutation
   of general relativity.

## The one-sentence version

> A physics prior buys extrapolation when the data can identify its
> parameters, and costs you accuracy when it cannot — and the only way to
> know which case you are in is to measure it against a control.

---

### Where to go next

* [`docs/gravity/`](../../docs/gravity) · [`docs/relativity/`](../../docs/relativity) · [`docs/quantum/`](../../docs/quantum) — the three problems in full
* [`docs/METHOD.md`](../../docs/METHOD.md) — the decisions, including the ones made *after* something went wrong
* [`docs/TOOLING.md`](../../docs/TOOLING.md) — how a formula comes out of symbolic regression
"""),
        ],
        "T6 - when PINNs fail",
    )


# ---------------------------------------------------------------------------
# T7 -- the regime where the prior actually wins
# ---------------------------------------------------------------------------


def t7_when_the_prior_wins():
    return _nb(
        [
            md("""
# T7 · When does a physics prior actually beat a black box?

[T6](T6_when_pinns_fail.ipynb) is uncomfortable reading: across four real
tracks the `pinn` arm wins **one** cell in twelve. That is not because
physics-informed learning does not work. It is because those tracks are the
wrong test — their laws are either **exact** (Kepler on a two-body system) or
**unidentifiable from the band observed** (Planck on FIRAS).

The case a physics prior was built for is the third one, and it was missing:

> The law is right as far as it goes, and **something real has been left out
> of it.**

That is the normal condition of applied physics — a neglected oblateness
term, a higher post-Newtonian order, an unmodelled instrument response. This
notebook builds exactly that, as a controlled experiment where the left-out
term is known and can be dialled.
"""),
            code(
                SETUP
                + """
from physprior.benchmark.neglected import NeglectedSystem, run_one, _fit_pinn
"""
            ),
            md(f"""
{section(1, "a law with a term removed")}

$$y(r) = \\underbrace{{\\frac{{GM}}{{r^{{2}}}}}}_{{\\text{{the law we model}}}}
\\;+\\; \\underbrace{{\\varepsilon\\,A\\,e^{{-((r-r_{{0}})/w)^{{2}}}}}}_{{\\text{{left out, never disclosed}}}}
\\;+\\; \\text{{noise}}$$

Three arms, and each can do something different about the missing piece:

| arm | what it can represent | what it cannot |
|---|---|---|
| `physics` | `GM` inside the law | the missing term, **at all** |
| `nn` | anything | — but must learn the whole curve from scratch |
| `pinn` | the law **and** a correction | — the network only has the residual to learn |

The prediction is a crossover. Let us see the system first.
"""),
            code("""
sys_ = NeglectedSystem(eps=0.4, noise=0.02, shape="bump")
r, y, clean = sys_.sample(40, seed=11)
grid = np.linspace(sys_.r_min, sys_.r_max, 400)

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.plot(grid, sys_.law(grid), lw=2.6, color=P.ARM_COLOR["physics"],
        label="the law we model,  GM/r^2")
ax.plot(grid, sys_.truth(grid), lw=2.0, color=P.INK_MUTED, ls=(0,(4,3)),
        label="the truth,  law + missing term")
ax.plot(r, y, "o", markersize=6, color=P.ARM_COLOR["pinn"],
        markeredgecolor=P.SURFACE, markeredgewidth=1.2, label="40 noisy samples")
ax.set_xlabel("r"); ax.set_ylabel("y")
ax.set_title("The missing term is small, smooth, and completely invisible here")
ax.legend(); plt.show()
print(f"the missing term averages {sys_.neglected_fraction*100:.1f}% of the law")
"""),
            md(f"""
{section(3, "what each loss actually is")}

`physics` minimises, over `GM` alone:

$$\\sum_i \\left(y_i - \\frac{{GM}}{{r_i^{{2}}}}\\right)^{{2}}$$

`nn` minimises, over the network weights alone:

$$\\sum_i \\left(y_i - \\mathrm{{NN}}(r_i)\\right)^{{2}}$$

`pinn` minimises, over **both at once**:

$$\\mathcal{{L}} = \\underbrace{{\\frac{{1}}{{\\sigma_y^{{2}}}}\\sum_i
\\left(y_i - \\frac{{GM}}{{r_i^{{2}}}} - \\sigma_y\\mathrm{{NN}}(r_i)\\right)^{{2}}}}_{{\\text{{data}}}}
\\;+\\; w_{{phys}}\\underbrace{{\\overline{{\\mathrm{{NN}}^{{2}}}}}}_{{\\text{{keep the correction small}}}}$$

The second term is the prior: *the law is nearly right, so the correction
should be nearly zero*. `w_phys` sets how strongly that is believed.
"""),
            md(f"""
{section(4, "the dial: how big is the missing term")}
"""),
            code("""
import pandas as pd
eps = pd.read_csv("results/neglected_eps.csv")
P.fig_neglected_sweep(eps, "eps", "size of the term left out of the law  (eps)",
                      "The physics prior pays as soon as the law is incomplete")
plt.show()
print(eps.groupby(["eps","arm"]).nrmse_in.median().unstack("arm")
        .to_string(float_format=lambda v: f"{v:.4g}"))
"""),
            md("""
**At `eps = 0` the law is exact and `physics` wins**, as it must — the PINN
pays 5× for a correction it does not need, and the black box pays 44×.

**From `eps = 0.1` onwards the PINN wins decisively**, and its error is
almost flat while `physics` degrades eight-fold. That is the crossover, and
it arrives as soon as there is *any* missing physics worth the name.

`nn` is flat at ~0.046 throughout. It never learns the curve well from forty
points, and the size of the missing term is irrelevant to it — it was
learning everything from scratch anyway.

### Did the network learn the missing physics, or just absorb noise?

This is the question that separates a useful correction from a flexible one.
The correction is plotted against the term it was **never shown**.
"""),
            code("""
_, gm, _, correction, hist = _fit_pinn(sys_, r, y, w_phys=0.01, epochs=2500, seed=11)
P.fig_learned_correction(sys_, correction); plt.show()
print(f"recovered GM = {gm:.4f}   (true 1.0)")
"""),
            md("""
It learned it. And because the correction carries the missing term, `GM`
comes back at **1.012** instead of being dragged off by physics it cannot
represent.

### What the loss was trading while that happened
"""),
            code("""
P.fig_learning_curves(hist, published=1.0,
                      title="What the loss traded, epoch by epoch")
plt.show()
"""),
            md("""
The data term falls thirtyfold. The physics term **rises** and plateaus —
that is the correction growing to the size of the missing bump and stopping
there, which is exactly what `w_phys` is negotiating. And the constant
overshoots to 1.038 before settling: *the loss going down and the constant
converging are different events.*
"""),
            md(f"""
{section(4, "noise, and where the advantage stops")}
"""),
            code("""
nz = pd.read_csv("results/neglected_noise.csv")
P.fig_neglected_sweep(nz, "noise", "measurement noise (fraction of signal spread)",
                      "...and stops paying when the correction starts fitting noise")
plt.show()
"""),
            md("""
A clean bias–variance crossover:

* `physics` is **biased but noise-immune** — it cannot fit the bump, and it
  cannot fit the noise either, so it sits flat at 0.055 whatever happens.
* `pinn` is **unbiased but not noise-immune**. Its correction is flexible
  enough to represent the missing term, which means it is flexible enough to
  represent noise, and above about 7% noise it starts doing so.

So the prior's advantage is **not unconditional**. It is a bias–variance
trade, and the crossover point is a measurable property of the problem rather
than a matter of taste.

### And the data budget
"""),
            code("""
bud = pd.read_csv("results/neglected_budget.csv")
P.fig_neglected_sweep(bud, "n_train", "training points",
                      "The black box needs data to reach a prior it never beats",
                      logx=True)
plt.show()
"""),
            md("""
`physics` is **flat**: it is bias-limited, and no amount of data fixes a
model that cannot represent the truth. `nn` improves elevenfold from 10 to
160 points and still does not catch the PINN. The prior is worth roughly a
factor of four in data here, and worth more the less data you have.
"""),
            md("""
## The catch, and the real conclusion

Everything above used a **localised** missing term — a bump the law has no
way to imitate. Repeat it with a missing term that looks like the law itself,
`eps·GM·R/r³`, one order higher in `1/r`, which is how a neglected oblateness
or first relativistic correction actually appears:
"""),
            code("""
deg = NeglectedSystem(eps=0.4, noise=0.02, shape="power")
rd, yd, _ = deg.sample(40, seed=11)
_, gm_deg, _, corr_deg, _ = _fit_pinn(deg, rd, yd, w_phys=0.01, epochs=2500, seed=11)
P.fig_learned_correction(deg, corr_deg,
    title="...and when it cannot: the law absorbs it instead")
plt.show()
print(f"distinguishable term : GM = {gm:.4f}   ({abs(gm-1)*100:.1f}% error)")
print(f"degenerate term      : GM = {gm_deg:.4f}   ({abs(gm_deg-1)*100:.1f}% error)")
"""),
            md("""
---

## The same question for a differential law

Everything above is algebraic, `y = law(x) + missing`. The other shape of
PINN solves a **differential equation**, and there the missing piece is a
missing **force** — so recovering it means the network has learned a term of
the equation of motion, not a curve.

The system is the one every physicist meets first. A pendulum obeys
$\\ddot\\theta = -\\omega^{2}\\sin\\theta$, and the small-angle step models it as
$\\ddot\\theta = -\\omega^{2}\\theta$. The residual PINN carries a learned force:

$$\\ddot\\theta + \\omega^{2}\\theta - C_\\phi(\\theta, \\dot\\theta) = 0$$

with $\\omega$ trainable, $C_\\phi$ penalised by `w_phys`, and $C_\\phi$
antisymmetrised so a force that does not vanish at rest at the origin is not
representable at all.

**And the same distinction decides everything.**
"""),
            code("""
import physprior.benchmark.neglected as N
ode = pd.read_csv("results/neglected_ode.csv").rename(columns={"nrmse": "nrmse_in"})
P.fig_neglected_sweep(ode[ode["shape"] == "damping"], "amplitude_deg",
    "pendulum amplitude (degrees)",
    "A conservative model cannot decay, at any omega")
plt.show()
print(ode[ode["shape"]=="damping"].groupby(["amplitude_deg","arm"]).nrmse_in.median()
        .unstack("arm").to_string(float_format=lambda v: f"{v:.4g}"))
"""),
            md("""
`physics` is pinned at **0.209 at every amplitude**. That is not a fit going
wrong — a conservative harmonic model *cannot produce decay at all*, at any
value of `ω`, so its error is a property of the model rather than of the
data. The PINN is 16–18× better as soon as there is enough amplitude to see.

### Did it learn the force itself?
"""),
            code("""
s_damp = N.NeglectedODE(amplitude=np.radians(60), noise=0.02, shape="damping")
t, th = s_damp.sample(60, seed=11)
ph, om_ph = N._ode_fit_physics(s_damp, t, th)
pr, om, _, force, hist = N._ode_fit_pinn(s_damp, t, th, w_phys=1e-3,
                                         epochs=2500, seed=11)
P.fig_learned_force(s_damp, force, pr, physics=ph); plt.show()
print(f"omega: physics {om_ph:.4f}, pinn {om:.4f}  (true 1.0)")
"""),
            md("""
It did — the learned force has the right sign and slope against **velocity**,
which is the variable it actually depends on. Plotting it against *angle*
would have drawn a flat line and told you nothing, which is the differential
version of asking the wrong question of the data.

### And now the degenerate case, again

Replace damping with the anharmonic term the small-angle step really drops,
$-\\omega^{2}(\\sin\\theta - \\theta)$. It depends on **angle**, and a pendulum at
amplitude $A$ has period $T(A)$ — so a harmonic oscillator can simply adopt
$\\omega = 2\\pi/T(A)$ and absorb most of it.
"""),
            code("""
P.fig_neglected_sweep(ode[ode["shape"] == "anharmonic"], "amplitude_deg",
    "pendulum amplitude (degrees)",
    "...but a shifted omega absorbs the anharmonic term")
plt.show()
s_anh = N.NeglectedODE(amplitude=np.radians(60), noise=0.02, shape="anharmonic")
ta, tha = s_anh.sample(60, seed=11)
pha, om_a = N._ode_fit_physics(s_anh, ta, tha)
pra, oma, _, fa, _ = N._ode_fit_pinn(s_anh, ta, tha, w_phys=1e-3, epochs=2500, seed=11)
P.fig_learned_force(s_anh, fa, pra, physics=pha); plt.show()
"""),
            md("""
No clean win, exactly as in the algebraic `1/r³` case. **The same principle
governs both forms of PINN:**

| | algebraic law | differential law |
|---|---|---|
| **degenerate** missing piece | `1/r³` — absorbed into `GM` | anharmonic — absorbed into `ω` |
| **distinguishable** missing piece | a localised bump | damping (depends on velocity) |
| result | prior wins by ~10× | prior wins by ~16× |

> A physics prior helps when the missing piece is **distinguishable from the
> law** — not merely when the law is incomplete. Which variable the missing
> term depends on is the whole question.

That is `quantum/cmb`'s identifiability finding, reached twice more from
completely different directions.
"""),
            md("""
---

## And once more, for a partial differential law

The last rung. The modelled law is pure diffusion, $u_t = \\alpha u_{xx}$, and
the truth carries one extra transport term. Here the degeneracy is not
approximate but **exact**, and it has a closed form.
"""),
            code("""
from physprior.benchmark.neglected import NeglectedPDE, _pde_fit_physics
base = _pde_fit_physics(NeglectedPDE(eps=0.0, noise=0.0, shape="diffusive"))
print(f"{'shape':<12}{'eps':>6}{'alpha_hat':>12}{'bias vs eps=0':>15}")
for shape in ("diffusive", "advective"):
    for eps in (0.0, 0.3, 0.6):
        s = NeglectedPDE(eps=eps, noise=0.0, shape=shape)
        a = _pde_fit_physics(s)
        print(f"{shape:<12}{eps:>6}{a:>12.5f}{(a/base-1)*100:>14.1f}%")
"""),
            md("""
**Two opposite failure signatures, from the same question.**

`diffusive` — the missing term is $\\varepsilon\\,\\alpha\\,u_{xx}$, *more of the
same operator*. A single rescaling $\\alpha \\to \\alpha(1+\\varepsilon)$
reproduces the truth exactly, so the fitted constant is wrong by **precisely
$\\varepsilon$** — and the prediction is flawless. This is the dangerous case:
nothing about the fit looks wrong.

`advective` — the missing term is $-v\\,u_x$, a drift. It is *orthogonal* to
$u_{xx}$ in the least-squares projection, so it does not bias $\\alpha$ at all.
Instead it makes the model wrong: diffusion is symmetric and no value of
$\\alpha$ can move a peak.

Look at what each model cannot reproduce, at its own best constant:
"""),
            code("""
for shape in ("diffusive", "advective"):
    s = NeglectedPDE(eps=0.3, noise=0.0, shape=shape)
    P.fig_pde_field(s, _pde_fit_physics(s)); plt.show()
"""),
            md("""
### One thing that did not work, reported as such

The 2-D residual PINN in this study **does not converge**, so its numbers are
marked rather than reported. The failure is unambiguous and it is worst where
it should be easiest — at `eps = 0`, where the modelled law is exactly right
and there is nothing whatever to recover, `alpha` is driven to ~0.001 against
a true 0.05 and the field is reproduced to only ~0.5 nRMSE.

**It is still not fixed**, after loss balancing and four further bug fixes.
The field error came down from 2.16 to 0.93 nRMSE and `alpha` is still 75%
wrong, so it stays marked rather than reported. What follows is the list of
what was actually wrong, because every item was a genuine defect and three of
them are mistakes this course warns about:

1. **No initial or boundary conditions at all**, so the residual did not
   identify `alpha` — many pairs `(u, alpha)` satisfy `u_t = alpha u_xx`.
2. **The residual was nondimensionalised by the amplitude, not the rate**,
   making the physics term ~1000× the data term. Since `u = u0` with
   `alpha = 0` gives an *exactly zero* residual, the optimiser took it.
3. **`_annealed_weight` was called with its arguments reversed.** Its
   signature is `(net, data, phys, ...)` and it returns
   `max|∇data| / mean|∇phys|`; passing the residual first returns the inverse
   ratio and *amplifies* precisely the term that was already too strong.
4. **The hard constraint used a bare `t`.** At the earliest data, `t = 0.02`
   and `x = 0.5`, the prefactor `t·x·(L−x)` is **0.005** — the network needed
   outputs of order 200 to correct anything. Replaced by a saturating
   `1 − exp(−t/τ)`, which vanishes at `t = 0` just as exactly.
5. **The initial profile was a piecewise-linear interpolant**, whose second
   derivative is zero almost everywhere — so the residual saw `u_xx` of the
   initial profile as *nothing*, when it is the largest term in the equation.
   That is [T1](T1_what_is_a_pinn.ipynb)'s ReLU warning, committed inside a
   hard constraint.
6. **A data-first warmup** before the residual is allowed to speak, because
   `alpha` is exactly the ratio `|u_t|/|u_xx|` and that ratio is meaningless
   until `u` is roughly right.

A control settles where the remaining problem is: a **plain MLP on the same
500 points reaches a field nRMSE of 0.095**, so the network can represent
this field easily. The constrained, physics-regularised version reaches 0.93.
The obstacle is therefore the parameterisation and the optimisation, not
capacity — and the next thing to try is the collocation sampling and a
curriculum in `t`, not more weight tuning.

So the PDE section above rests entirely on the `physics` arm — whose result
is a closed-form identity and needs no network at all, so the finding
does not depend on the thing that failed.

What it would take is not a parameter tweak. It is the Phase 2 machinery —
gradient-norm loss balancing, measured in [T6](T6_when_pinns_fail.ipynb) to
be worth up to 101× on the 1-D tracks — applied to a 2-D residual, where the
data and physics terms differ by three orders of magnitude at initialisation.
That is exactly the disease balancing treats, and it is the obvious next
piece of work.

The advective residual is a **dipole** — mass moved from one side to the
other, which is exactly what a drift does and exactly what diffusion cannot.
Its peak is six times the diffusive one.

So the arc closes, and the same principle governs all three rungs:

| law | degenerate missing piece | distinguishable missing piece |
|---|---|---|
| **algebraic** `y = GM/r²` | `1/r³` → absorbed into `GM` | a localised bump |
| **ODE** `θ̈ = -ω²θ` | anharmonic → absorbed into `ω` | damping (velocity) |
| **PDE** `u_t = α u_xx` | `ε α u_xx` → absorbed into `α`, **exactly** | advection (drift) |

> The question is never "is the law incomplete?" It is **"can the model's
> free parameters imitate what is missing?"** If they can, the fit looks
> perfect and the constant is quietly wrong. If they cannot, the residual has
> structure and a physics prior has something to learn.
"""),
            md("""
Over a finite range of `r`, `1/r³` is **nearly degenerate with `1/r²`**:
raising `GM` mimics most of it. So the fit absorbs the missing physics into
the constant, the correction learns nothing identifiable, and `GM` comes back
**27% wrong** instead of 1%.

> **A physics prior does not help because the law is incomplete. It helps
> when the missing piece is *distinguishable from the law*.**

That is the same lesson as `quantum/cmb`, where the Planck denominator is not
identifiable from the Wien-dominated band FIRAS observed — arriving here from
a completely different direction, in a system where the answer is known
exactly.

It also explains T6's scorecard. The `pinn` arm wins one cell
in twelve on the real tracks **because those tracks are mostly the exact-law
and the degenerate cases**. Given a track in the third regime, it wins by an
order of magnitude.
"""),
        ],
        "T7 - when the prior wins",
    )


TUTORIALS = {
    "T1_what_is_a_pinn": t1_what_is_a_pinn,
    "T2_forward_and_inverse": t2_forward_and_inverse,
    "T3_gravity": t3_gravity,
    "T4_relativity": t4_relativity,
    "T5_quantum_wavefunction": t5_quantum_wavefunction,
    "T6_when_pinns_fail": t6_when_pinns_fail,
    "T7_when_the_prior_wins": t7_when_the_prior_wins,
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
    out = settings.notebooks_dir / "tutorials"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in TUTORIALS.items():
        if only and only.lower() not in name.lower():
            continue
        nb = fn()
        path = out / f"{name}.ipynb"
        nbf.write(nb, path)
        print(f"wrote {_display(path, settings.root)}")
        if execute:
            from nbclient import NotebookClient

            print("  executing ...", flush=True)
            NotebookClient(
                nb,
                timeout=3600,
                kernel_name="physprior",
                resources={"metadata": {"path": str(settings.root)}},
                allow_errors=True,
            ).execute()
            nbf.write(nb, path)
            errs = [
                c
                for c in nb.cells
                if any(o.get("output_type") == "error" for o in c.get("outputs", []))
            ]
            print(f"  done, {len(errs)} cell(s) with errors")
            for c in errs:
                for o in c.get("outputs", []):
                    if o.get("output_type") == "error":
                        print(f"    {o.get('ename')}: {str(o.get('evalue'))[:120]}")


if __name__ == "__main__":
    build(
        execute="--execute" in sys.argv,
        only=next((a for a in sys.argv[1:] if not a.startswith("-")), None),
    )
