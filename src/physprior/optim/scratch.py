"""A multilayer perceptron in plain NumPy, with backpropagation written out.

Teaching code. Everything the `nn` and `pinn` arms get from torch is done
here by hand, so each step can be read and checked:

    forward      a_0 = x;  z_l = a_{l-1} W_l + b_l;  a_l = act(z_l);  y = z_L
    loss         MSE = mean((y - t)^2)
    backward     the chain rule, layer by layer, from the output back
    optimizers   SGD, heavy-ball momentum, Nesterov, RMSprop, Adam

The gradient is checked twice: against central finite differences, and
against torch autograd on an identical network (tests/test_optim_scratch.py).

Initialisation. A layer's pre-activation variance is fan_in * Var(W) *
Var(input). Keeping it near one through depth needs Var(W) ~ 1/fan_in.
Xavier/Glorot (2010) uses 2/(fan_in + fan_out), which also keeps the
backward pass stable, and suits tanh (odd, unit slope at zero). He (2015)
uses 2/fan_in because ReLU zeroes half its inputs and so halves the variance.
Too-large weights saturate tanh and the gradient vanishes; too-small weights
shrink the signal geometrically with depth.

Activation. PINNs use tanh because the loss contains derivatives of the
network with respect to its input. tanh is smooth, so d^2u/dx^2 exists and is
itself smooth. ReLU's second derivative is zero almost everywhere, so a ReLU
network cannot represent u'' = f at all: the residual of any second-order ODE
sees only the network's piecewise-linear pieces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# activations: value and derivative
# ---------------------------------------------------------------------------


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def dtanh(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def drelu(z: np.ndarray) -> np.ndarray:
    # The derivative at exactly zero is a convention; 0 matches torch.
    return (z > 0).astype(float)


ACTIVATIONS = {"tanh": (tanh, dtanh), "relu": (relu, drelu)}


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass
class MLP:
    """Weights W[l] of shape (fan_in, fan_out) and biases b[l] of (fan_out,)."""

    W: list[np.ndarray]
    b: list[np.ndarray]
    activation: str = "tanh"

    @property
    def sizes(self) -> list[int]:
        return [self.W[0].shape[0]] + [w.shape[1] for w in self.W]

    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b, strict=True))

    # Optimizers work on one flat vector; the network on per-layer arrays.
    def flat(self) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate([w.ravel(), b])
                for w, b in zip(self.W, self.b, strict=True)
            ]
        )

    def set_flat(self, theta: np.ndarray) -> None:
        k = 0
        for i, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
            self.W[i] = theta[k : k + w.size].reshape(w.shape).copy()
            k += w.size
            self.b[i] = theta[k : k + b.size].copy()
            k += b.size


def init_mlp(
    sizes: list[int],
    activation: str = "tanh",
    scheme: str | None = None,
    seed: int = 0,
) -> MLP:
    """`scheme` defaults to the one matched to the activation: xavier for
    tanh, he for relu. Biases start at zero."""
    rng = np.random.default_rng(seed)
    scheme = scheme or ("he" if activation == "relu" else "xavier")
    W, b = [], []
    for fan_in, fan_out in itertools.pairwise(sizes):
        if scheme == "xavier":
            sd = np.sqrt(2.0 / (fan_in + fan_out))
        elif scheme == "he":
            sd = np.sqrt(2.0 / fan_in)
        elif scheme == "naive":  # unit variance: saturates tanh at depth
            sd = 1.0
        else:
            raise ValueError(f"unknown init scheme {scheme!r}")
        W.append(rng.normal(0.0, sd, (fan_in, fan_out)))
        b.append(np.zeros(fan_out))
    return MLP(W, b, activation)


# ---------------------------------------------------------------------------
# forward, loss, backward
# ---------------------------------------------------------------------------


def forward(net: MLP, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Returns the output (N, d_out) and the cache backprop needs: every
    layer's input `a` and pre-activation `z`."""
    act, _ = ACTIVATIONS[net.activation]
    a = np.atleast_2d(x)
    cache: dict[str, list[np.ndarray]] = {"a": [a], "z": []}
    last = len(net.W) - 1
    for i, (w, b) in enumerate(zip(net.W, net.b, strict=True)):
        z = a @ w + b
        cache["z"].append(z)
        # The output layer is linear: a regression target is not bounded.
        a = z if i == last else act(z)
        cache["a"].append(a)
    return a, cache


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """The loss and its gradient with respect to the prediction."""
    r = pred - target.reshape(pred.shape)
    return float(np.mean(r**2)), 2.0 * r / r.size


def backward(
    net: MLP, cache: dict, dloss_dy: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagation.

    With delta_l = dL/dz_l:
        output layer   delta_L = dL/dy                (linear output)
        each layer     dL/dW_l = a_{l-1}^T delta_l,   dL/db_l = sum_rows delta_l
        going down     delta_{l-1} = (delta_l W_l^T) * act'(z_{l-1})
    """
    _, dact = ACTIVATIONS[net.activation]
    n_layers = len(net.W)
    gW: list[np.ndarray] = [np.empty(0)] * n_layers
    gb: list[np.ndarray] = [np.empty(0)] * n_layers
    delta = dloss_dy
    for i in reversed(range(n_layers)):
        gW[i] = cache["a"][i].T @ delta
        gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ net.W[i].T) * dact(cache["z"][i - 1])
    return gW, gb


def loss_and_grad(net: MLP, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """MSE and its gradient as one flat vector, in `MLP.flat` order."""
    pred, cache = forward(net, x)
    loss, dy = mse(pred, y)
    gW, gb = backward(net, cache, dy)
    g = np.concatenate(
        [np.concatenate([w.ravel(), b]) for w, b in zip(gW, gb, strict=True)]
    )
    return loss, g


# ---------------------------------------------------------------------------
# gradient checks
# ---------------------------------------------------------------------------


def finite_difference_grad(
    net: MLP, x: np.ndarray, y: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """Central differences, error O(h^2). One pair of forward passes per
    parameter, which is why nobody trains this way."""
    theta = net.flat()
    g = np.empty_like(theta)
    for k in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        net.set_flat(tp)
        lp = mse(forward(net, x)[0], y)[0]
        net.set_flat(tm)
        lm = mse(forward(net, x)[0], y)[0]
        g[k] = (lp - lm) / (2.0 * h)
    net.set_flat(theta)
    return g


def fd_convergence(
    net: MLP, x: np.ndarray, y: np.ndarray, steps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
) -> list[dict]:
    """The finite-difference check's own convergence study: the error against
    backprop should fall as h^2 until round-off (~eps/h) takes over."""
    _, g = loss_and_grad(net, x, y)
    rows = []
    for h in steps:
        gfd = finite_difference_grad(net, x, y, h)
        rows.append(
            {"h": h, "rel_error": float(np.linalg.norm(gfd - g) / np.linalg.norm(g))}
        )
    return rows


def torch_grad(net: MLP, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The same network in torch, differentiated by autograd."""
    import torch

    x_t = torch.tensor(np.atleast_2d(x), dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64).reshape(len(x_t), -1)
    Ws = [torch.tensor(w, requires_grad=True) for w in net.W]
    bs = [torch.tensor(b, requires_grad=True) for b in net.b]
    act = torch.tanh if net.activation == "tanh" else torch.relu
    a = x_t
    for i, (w, b) in enumerate(zip(Ws, bs, strict=True)):
        z = a @ w + b
        a = z if i == len(Ws) - 1 else act(z)
    loss = torch.mean((a - y_t) ** 2)
    grads = torch.autograd.grad(loss, [*Ws, *bs])
    gW, gb = grads[: len(Ws)], grads[len(Ws) :]
    return np.concatenate(
        [
            np.concatenate([w.numpy().ravel(), b.numpy()])
            for w, b in zip(gW, gb, strict=True)
        ]
    )


# ---------------------------------------------------------------------------
# optimizers, by hand
# ---------------------------------------------------------------------------
#
# Each takes the flat parameter vector and its gradient and returns the new
# vector. State (velocities, moment estimates) lives on the object.


@dataclass
class SGD:
    """theta <- theta - lr * g. The step is proportional to the gradient, so
    it is too large along sharp directions and too small along flat ones;
    the stable learning rate is capped at 2 / lambda_max of the Hessian."""

    lr: float

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        return theta - self.lr * g


@dataclass
class Momentum:
    """Heavy ball (Polyak 1964): v <- beta v - lr g;  theta <- theta + v.
    Velocity accumulates along directions where the gradient keeps its sign
    and cancels where it oscillates, which damps the zig-zag across a narrow
    valley."""

    lr: float
    beta: float = 0.9
    v: np.ndarray | None = field(default=None, repr=False)

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v - self.lr * g
        return theta + self.v


@dataclass
class Nesterov:
    """Nesterov (1983): take the gradient at the look-ahead point theta + beta v.
    Written in the equivalent form that needs the gradient at theta only
    (Sutskever et al. 2013, and torch's `nesterov=True`):
        v <- beta v - lr g;   theta <- theta + beta v - lr g
    """

    lr: float
    beta: float = 0.9
    v: np.ndarray | None = field(default=None, repr=False)

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v - self.lr * g
        return theta + self.beta * self.v - self.lr * g


@dataclass
class RMSprop:
    """Divide each coordinate's step by a running RMS of its gradient
    (Tieleman & Hinton 2012). A diagonal rescaling: it equalises step sizes
    across parameters whose gradients differ by orders of magnitude, which is
    exactly the situation in a PINN with a physical constant next to network
    weights."""

    lr: float
    rho: float = 0.99
    eps: float = 1e-8
    s: np.ndarray | None = field(default=None, repr=False)

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.rho * self.s + (1.0 - self.rho) * g**2
        return theta - self.lr * g / (np.sqrt(self.s) + self.eps)


@dataclass
class Adam:
    """Momentum on the gradient plus RMSprop's rescaling (Kingma & Ba 2015).

    Both running averages start at zero, so early on they are biased toward
    zero by factors (1 - beta1^t) and (1 - beta2^t). Dividing those out is the
    bias correction. Without it the first step is m/sqrt(v) =
    (1 - beta1)/sqrt(1 - beta2) ~ 3.2 times lr for the default betas: the
    second-moment estimate is the more biased of the two, so the uncorrected
    optimizer starts with steps three times larger than intended. With the
    correction the first step has magnitude lr in every coordinate.
    """

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    bias_correction: bool = True
    m: np.ndarray | None = field(default=None, repr=False)
    v: np.ndarray | None = field(default=None, repr=False)
    t: int = 0

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g**2
        m_hat, v_hat = self.m, self.v
        if self.bias_correction:
            m_hat = self.m / (1.0 - self.beta1**self.t)
            v_hat = self.v / (1.0 - self.beta2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def make_optimizer(name: str, lr: float):
    return OPTIMIZERS[name](lr=lr)


# ---------------------------------------------------------------------------
# training, and a physics curve to train on
# ---------------------------------------------------------------------------


def damped_oscillation(t: np.ndarray, gamma: float = 0.3, omega: float = 2.0):
    """x(t) = exp(-gamma t) cos(omega t): a damped oscillator released from
    rest-ish. Not the exact ODE solution for x'(0) = 0 (that has a sine term);
    it is a smooth 1-D physics curve for the network to fit."""
    return np.exp(-gamma * t) * np.cos(omega * t)


def demo_data(n: int = 64, seed: int = 0, noise: float = 0.0):
    """Inputs rescaled to [-1, 1]: a network trains on O(1) inputs, not on
    physical units (the repo's `Standardiser` does the same)."""
    t = np.linspace(0.0, 6.0, n)
    y = damped_oscillation(t)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, n)
    x = (t / 3.0 - 1.0).reshape(-1, 1)
    return x, y.reshape(-1, 1), t


def train(
    net: MLP,
    x: np.ndarray,
    y: np.ndarray,
    optimizer,
    steps: int = 3000,
    record_every: int = 10,
) -> dict[str, list]:
    """Full-batch training. Returns the loss curve; stops early on divergence."""
    theta = net.flat()
    hist: dict[str, list] = {"step": [], "loss": []}
    for k in range(steps + 1):
        net.set_flat(theta)
        loss, g = loss_and_grad(net, x, y)
        if not np.isfinite(loss) or loss > 1e8:
            hist["step"].append(k)
            hist["loss"].append(float("inf"))
            break
        if k % record_every == 0 or k == steps:
            hist["step"].append(k)
            hist["loss"].append(loss)
        if k < steps:
            theta = optimizer.step(theta, g)
    net.set_flat(theta)
    return hist


def compare_optimizers(
    lrs: dict[str, float] | None = None,
    sizes=(1, 32, 32, 1),
    steps: int = 3000,
    seed: int = 0,
) -> dict[str, dict]:
    """Every hand-written optimizer from the same initial network.

    Default learning rates are one sensible value per method, not a tuned
    optimum; `physprior.optim.optimizers_study` does the grid search.
    """
    lrs = lrs or {
        "sgd": 0.1,
        "momentum": 0.02,
        "nesterov": 0.02,
        "rmsprop": 1e-3,
        "adam": 3e-3,
    }
    x, y, _ = demo_data()
    out = {}
    for name, lr in lrs.items():
        net = init_mlp(list(sizes), "tanh", seed=seed)
        out[name] = {
            "lr": lr,
            "history": train(net, x, y, make_optimizer(name, lr), steps),
        }
        out[name]["net"] = net
    return out
