"""Two tutorials on optimization, generated as notebooks.

    T8  a network from scratch: forward, backprop, gradient check, SGD to Adam
    T9  optimizers and loss functions: physics versus black box

T8 runs everything live (NumPy only, plus torch for one check). T9 runs two
small live demonstrations and reads the rest from results/optim, which
`physprior.optim.report.run_all()` writes.
"""

from __future__ import annotations

from physprior.reporting.cells import SETUP, _nb, code, md

# ---------------------------------------------------------------------------
# T8
# ---------------------------------------------------------------------------


def t8_network_from_scratch():
    return _nb(
        [
            md("""
# T8 · A network from scratch: forward, backprop, gradient check, SGD to Adam

Every arm in this project that contains a network gets its gradients from
torch. This notebook does the same job by hand, in NumPy, so that each step
can be read and checked: the forward pass, the backward pass, a gradient check
against finite differences and against torch, the choice of initialisation,
and five optimizers from plain SGD to Adam.

The code lives in `physprior.optim.scratch`. It is short on purpose.
"""),
            code(SETUP),
            code("""
from physprior.optim import scratch as S

net = S.init_mlp([1, 16, 16, 1], activation="tanh", seed=0)
x, y, t = S.demo_data(64)          # a damped oscillation, inputs scaled to [-1, 1]
out, cache = S.forward(net, x)
print("layer sizes", net.sizes, "  parameters", net.n_params())
print("output", out.shape, "  cached activations", [a.shape for a in cache["a"]])
"""),
            md("""
## 1. Forward and backward

Forward: `a_0 = x`, `z_l = a_{l-1} W_l + b_l`, `a_l = tanh(z_l)`, and the last
layer is linear. The loss is `L = mean((y_pred - y)^2)`.

Backward, with `delta_l = dL/dz_l`:

- output layer: `delta_L = dL/dy_pred = 2 (y_pred - y) / N`
- each layer: `dL/dW_l = a_{l-1}^T delta_l`, `dL/db_l = sum over rows of delta_l`
- one layer down: `delta_{l-1} = (delta_l W_l^T) * tanh'(z_{l-1})`, with
  `tanh' = 1 - tanh^2`

That is the whole of backpropagation: the chain rule applied once per layer,
reusing what the forward pass cached. The cost is about two forward passes,
whatever the number of parameters.
"""),
            code("""
loss, g = S.loss_and_grad(net, x, y)
g_torch = S.torch_grad(net, x, y)
print(f"loss {loss:.4f}   |grad| {np.linalg.norm(g):.4f}")
print(f"relative difference, backprop vs torch autograd: "
      f"{np.linalg.norm(g - g_torch) / np.linalg.norm(g):.1e}")
"""),
            md("""
## 2. The gradient check, and its own convergence

A central difference `(L(theta + h e_k) - L(theta - h e_k)) / 2h` has
truncation error O(h^2) and round-off error ~ eps / h. So the check is itself
a numerical method with a convergence study: the error against backprop
should fall with slope 2 on a log-log plot until round-off takes over.
"""),
            code("""
rows = S.fd_convergence(net, x, y, steps=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))
hs = np.array([r["h"] for r in rows]); errs = np.array([r["rel_error"] for r in rows])
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.loglog(hs, errs, "o-", color=P.ARM_COLOR["physics"], label="backprop vs central difference")
ax.loglog(hs, errs[1] * (hs / hs[1])**2, ":", color=P.INK_2, label="slope 2")
ax.set_xlabel("h"); ax.set_ylabel("relative error"); ax.legend()
plt.show()
for r in rows: print(f"h = {r['h']:.0e}   error = {r['rel_error']:.2e}")
"""),
            md("""
## 3. Initialisation

The variance of a layer's pre-activation is `fan_in * Var(W) * Var(input)`.
To keep it near one through depth, `Var(W)` must scale like `1/fan_in`.
Xavier uses `2 / (fan_in + fan_out)` and suits tanh; He uses `2 / fan_in`
because ReLU zeroes half its inputs. Below: eight layers of width 64.
"""),
            code("""
from physprior.optim.report import init_depth_scan
scan = init_depth_scan()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, act in zip(axes, ["tanh", "relu"]):
    for scheme, col in [("xavier", "#2a78d6"), ("he", "#eb6834"), ("naive", "#4a3aa7")]:
        d = scan[(scan.activation == act) & (scan.init == scheme)]
        ax.semilogy(d.layer, d["std"], "o-", color=col, label=scheme)
    ax.set_title(act); ax.set_xlabel("layer"); ax.set_ylabel("std of activations"); ax.legend()
plt.show()
print(scan[scan.layer == scan.layer.max()].to_string(index=False))
"""),
            md("""
With unit-variance weights the tanh network's units sit in the flat tails
(saturated: almost no gradient passes), and the ReLU network's activations
grow by orders of magnitude. The matched schemes keep the scale within a
factor of a few.

## 4. Why PINNs use tanh

A PINN's loss contains derivatives of the network with respect to its input.
A ReLU network is piecewise linear, so its second derivative is zero except at
the kinks. It cannot satisfy `u'' = -omega^2 u` anywhere.
"""),
            code("""
xs = np.linspace(-1, 1, 2001).reshape(-1, 1); h = xs[1, 0] - xs[0, 0]
fig, ax = plt.subplots(figsize=(6, 3.4))
for act, col in [("tanh", "#2a78d6"), ("relu", "#eb6834")]:
    n = S.init_mlp([1, 32, 32, 1], act, seed=1)
    u = S.forward(n, xs)[0].ravel()
    upp = (u[2:] - 2 * u[1:-1] + u[:-2]) / h**2
    ax.plot(xs[1:-1, 0], upp, color=col, lw=1.2, label=f"{act}: d2u/dx2")
    print(f"{act}: fraction of points with |u''| < 1e-6: {np.mean(np.abs(upp) < 1e-6):.2f}")
ax.set_ylim(-20, 20); ax.legend(); ax.set_xlabel("x")
plt.show()
"""),
            md("""
## 5. Optimizers, by hand

Each optimizer maps (parameters, gradient) to new parameters:

- **SGD** `theta -= lr g`. On a quadratic with Hessian eigenvalue lambda the
  error along that direction is multiplied by `(1 - lr lambda)` per step, so
  SGD is stable only for `lr < 2 / lambda_max`, and then crawls along the
  directions with small lambda.
- **Momentum** (heavy ball) accumulates a velocity: consistent gradients add
  up, oscillating ones cancel.
- **Nesterov** evaluates the gradient at the look-ahead point.
- **RMSprop** divides each coordinate by a running RMS of its gradient: a
  diagonal preconditioner.
- **Adam** is momentum plus RMSprop's scaling, with bias correction for the
  zero-initialised running averages.
"""),
            code("""
A = np.diag([1.0, 50.0]); lam_max = 50.0
fig, ax = plt.subplots(figsize=(6, 3.4))
for lr, col in [(0.9 * 2 / lam_max, "#2a78d6"), (1.05 * 2 / lam_max, "#eb6834")]:
    th = np.array([1.0, 1.0]); opt = S.SGD(lr); f = []
    for _ in range(60):
        f.append(0.5 * th @ A @ th); th = opt.step(th, A @ th)
    ax.semilogy(f, color=col, label=f"lr = {lr * lam_max / 2:.2f} x 2/lambda_max")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); plt.show()
"""),
            code("""
res = S.compare_optimizers(steps=2000)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {"sgd": "#2a78d6", "momentum": "#eb6834", "rmsprop": "#1baf7a", "adam": "#4a3aa7", "nesterov": P.INK}
for k, v in res.items():
    h = v["history"]
    axes[0].semilogy(h["step"], h["loss"], color=cols[k], ls="--" if k == "nesterov" else "-",
                     label=f"{k} (lr {v['lr']:g})")
axes[0].set_xlabel("step"); axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=8)
axes[1].plot(t, y, "o", color=P.INK_MUTED, ms=4, label="data")
for k in ("sgd", "adam"):
    axes[1].plot(t, S.forward(res[k]["net"], x)[0].ravel(), color=cols[k], label=k)
axes[1].set_xlabel("t"); axes[1].legend(); plt.show()
for k, v in res.items(): print(f"{k:9s} final MSE {v['history']['loss'][-1]:.2e}")
"""),
            md("""
The SGD curve shows spikes: its rate sits near the stability limit of the
sharpest direction, and when training sharpens the loss past `2 / lr` it
overshoots and recovers (the edge of stability, Cohen et al. 2021). Adam
spikes too, for the same reason in its own rescaled geometry, but its
per-coordinate scaling lets it take large steps along flat directions, which
is where SGD loses most of its time.

### Adam's bias correction

Both running averages start at zero. At step 1, `m = 0.1 g` and
`v = 0.001 g^2`, so the uncorrected step is `0.1 / sqrt(0.001) = 3.2` times
the learning rate. Dividing by `(1 - beta^t)` removes that.
"""),
            code("""
g0 = np.array([1e-3, 1.0, -50.0])
print("with correction   ", S.Adam(lr=0.01).step(np.zeros(3), g0))
print("without correction", S.Adam(lr=0.01, bias_correction=False).step(np.zeros(3), g0))
"""),
            md("""
## Where this goes next

T9 takes these optimizers to the project's problems and asks whether a
physics model, a PINN and a black box want the same optimizer. They do not,
and the Hessian at the solution explains why.
"""),
        ],
        "T8 · A network from scratch",
    )


# ---------------------------------------------------------------------------
# T9
# ---------------------------------------------------------------------------


def t9_optimizers_and_losses():
    return _nb(
        [
            md("""
# T9 · Optimizers and loss functions: physics versus black box

Three model families on the same problems (`physprior.optim`):

- `physics`: the closed-form law with 1-3 constants trainable;
- `pinn`: law plus a penalised network correction (hydrogen), or the
  residual PINN whose network is the solution of the ODE/PDE (oscillator,
  heat equation);
- `nn`: a tanh MLP, width 32, depth 3.

The full study (`physprior.optim.report.run_all()`) writes `results/optim/`.
This notebook runs two small live examples and then reads those results.
Learning rates were chosen on the tuning seeds 3/7/19; reported numbers are on
11/23/42.
"""),
            code(SETUP),
            md("""
## 1. Live: one problem, three models, two optimizers

The damped oscillator `x'' + 2 gamma x' + omega0^2 x = 0` with gamma and
omega0 unknown. Adam against L-BFGS on each model, and Levenberg-Marquardt on
the physics model, counting loss-and-gradient evaluations.
"""),
            code("""
import pandas as pd
from physprior.optim import optimizers_study as O
rows = []
for model in ("physics", "pinn", "nn"):
    for opt, lr in (("adam", 1e-2), ("lbfgs", 1.0)):
        r, _, _ = O.run_one(O.RunSpec("oscillator", model, opt, lr, 11, 600))
        rows.append(r)
rows.append(O.run_lm("oscillator", 11))
live = pd.DataFrame(rows)
live[["model", "optimizer", "n_params", "evals_used", "evals_to_tol",
      "final_data_loss", "nrmse_in", "nrmse_out", "param_err_pct"]]
"""),
            md("""
## 2. Live: the Hessian at the solution

The physics model has two parameters, so its Hessian is 2 x 2. The black box
has about 2200, and most of its eigenvalues are zero to round-off.
"""),
            code("""
import torch
from physprior.optim import problems as Pb
task = Pb.oscillator()
for kind in ("physics", "nn"):
    m = Pb.Model(task, kind, seed=11)
    opt = torch.optim.LBFGS(m.parameters(), max_iter=400, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = m.loss(); l.backward(); return l
    opt.step(closure)
    eig = np.linalg.eigvalsh(Pb.full_hessian(m.loss, m.parameters()))[::-1]
    lmax = eig[0]
    n_sharp = int(np.sum(eig > 1e-3 * lmax))
    flat = float(np.mean(np.abs(eig) < 1e-6 * lmax))
    print(f"{kind:8s} params {eig.size:5d}  lambda_max {lmax:.3g}  "
          f"sharp (> 1e-3 lambda_max) {n_sharp}  flat fraction {flat:.3f}  "
          f"lambda_max / lambda_min {lmax / eig[-1]:.3g}")
"""),
            md("""
For the network, a negative `lambda_max / lambda_min` means the smallest
eigenvalue is negative: the end point is not a strict minimum, and the ratio
is not a condition number. Most of its eigenvalues are zero to within 1e-6 of
the largest, which is what "flat directions" means in practice.

## 3. The full study: learning-rate sensitivity and training curves
"""),
            code("""
from IPython.display import Image, display, Markdown
from physprior.config import get_settings
from physprior.io import load_table
fig_dir = get_settings().figures_dir / "optim"
def show(name):
    p = fig_dir / f"{name}.png"
    display(Image(str(p)) if p.exists() else Markdown(f"*{name}: run the study first*"))
show("03_lr_sensitivity"); show("04_loss_curves")
"""),
            code("""
from physprior.optim.optimizers_study import summarise
s = summarise(load_table("optim", "optimizers"))
s[["task", "model", "optimizer", "lr", "n_params", "evals_to_tol", "failure_rate",
   "final_data_loss", "nrmse_in", "nrmse_out", "param_err_pct"]]
"""),
            md("""
## 4. Curvature and the PINN's two loss terms

Top: the Hessian spectrum at the Adam solution. Bottom: the ratio of the
gradient norms of the physics and data terms over the PINN's network weights
during training. A ratio far from one means one term steers the shared
weights and the other is nearly invisible to the optimizer (Wang, Teng &
Perdikaris 2021).
"""),
            code("""
show("05_hessian_spectra"); show("06_grad_ratio")
from physprior.io import load_json
from physprior.optim.report import curvature_table
curvature_table(load_table("optim", "curvature"), load_json("optim", "hessian_eigenvalues"))
"""),
            md("""
## 5. Loss functions under three kinds of noise

MSE is the Gaussian likelihood. With heavy-tailed noise or outliers it gives
every large residual a quadratic pull, which biases a fitted constant; L1,
Huber and Cauchy cap that pull. The black box has no constant to bias, but its
curve bends toward the outliers.
"""),
            code("""
show("07_loss_functions")
ls = load_table("optim", "losses_summary")
ls[ls.model == "physics"][["noise", "loss", "gamma_bias_pct", "gamma_sd_pct", "gamma_bias_z",
                           "nrmse_in", "nrmse_out"]]
"""),
            code("""
show("08_w_phys_dial")
load_table("optim", "w_phys_dial").groupby(["noise", "w_phys"])[
    ["nrmse_in", "nrmse_out", "err_gamma_pct"]].median()
"""),
            md("""
## 6. The frozen `balance` option

The shipped PINN anneals its physics weight from the ratio of gradient norms.
For the shape-B PINN the physics term is `mean(NN^2)`, whose gradient is
proportional to the correction itself, so the weight rises as the correction
shrinks and the correction shrinks as the weight rises. Measured on the real
tracks, tuning seeds only:
"""),
            code("""
show("09_balance_history"); show("10_balance_errors")
load_table("optim", "balance_summary")[
    ["track", "split", "variant", "w_phys_final", "correction_rms_frac",
     "correction_rms_frac_out", "nrmse_in", "nrmse_out", "nrmse_out_vs_physics"]]
"""),
            md("""
The written conclusions, with every number read from these tables, are in
`docs/optimization/README.md`.
"""),
        ],
        "T9 · Optimizers and loss functions",
    )


TUTORIALS_OPTIM = {
    "T8_network_from_scratch": t8_network_from_scratch,
    "T9_optimizers_and_losses": t9_optimizers_and_losses,
}
