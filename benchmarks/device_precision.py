"""What float32 costs a second-derivative quantity.

MPS has no float64 kernels, so choosing the GPU here chooses single
precision. This measures the price on the ANALYTIC field, where the answer
is known exactly (alpha = 0.05) and the network cannot be blamed.
"""
import torch

L, W, ALPHA, T_MAX = 1.0, 0.08, 0.05, 1.0


def implied_alpha(dtype, dev, n=200_000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(n, generator=g, dtype=torch.float64) * L).to(
        device=dev, dtype=dtype).requires_grad_(True)
    t = (torch.rand(n, generator=g, dtype=torch.float64) * T_MAX).to(
        device=dev, dtype=dtype).requires_grad_(True)
    s2 = W**2 + 4.0 * ALPHA * t
    u = (W / torch.sqrt(s2)) * torch.exp(-((x - 0.5 * L) ** 2) / s2)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_t, u_xx = u_t.detach().cpu().double(), u_xx.detach().cpu().double()
    return float((u_t * u_xx).sum() / (u_xx**2).sum()), u_xx


rows = [
    ("cpu  float64", torch.float64, torch.device("cpu")),
    ("cpu  float32", torch.float32, torch.device("cpu")),
    ("mps  float32", torch.float32, torch.device("mps")),
]
print(f"{'':14} {'implied alpha':>20} {'rel err':>12} {'worst pointwise u_xx':>22}")
ref = None
for label, dtype, dev in rows:
    a, u_xx = implied_alpha(dtype, dev)
    if ref is None:
        ref = u_xx
        worst = "-- (reference)"
    else:
        m = u_xx.abs().max()
        worst = f"{float(((u_xx - ref).abs() / m).max()):.3e}"
    print(f"{label:14} {a:>20.12f} {abs(a / ALPHA - 1):>11.2e} {worst:>22}")
