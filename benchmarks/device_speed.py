"""The PDE PINN inner loop, on each device, at each precision.

Deliberately a replica rather than an import: this venv exists only to reach
MPS on an OS the project's own torch has dropped, and nothing here writes
into the project.
"""
import time
import torch

L, T_MAX, W, ALPHA = 1.0, 1.0, 0.08, 0.05


def mlp(nin, width, depth, dtype, dev):
    layers, d = [], nin
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers += [torch.nn.Linear(d, 1)]
    return torch.nn.Sequential(*layers).to(device=dev, dtype=dtype)


def step_time(dev, dtype, n_coll, epochs=60, width=48, depth=4):
    torch.manual_seed(11)
    net = mlp(2, width, depth, dtype, dev)
    corr = mlp(2, width, depth, dtype, dev)
    log_a = torch.nn.Parameter(torch.tensor(-3.0, dtype=dtype, device=dev))
    opt = torch.optim.Adam([*net.parameters(), *corr.parameters(), log_a], lr=3e-3)

    def run(k):
        for _ in range(k):
            opt.zero_grad()
            x = torch.rand(n_coll, dtype=dtype, device=dev, requires_grad=True) * L
            t = torch.rand(n_coll, dtype=dtype, device=dev, requires_grad=True) * T_MAX
            gate = 1.0 - torch.exp(-t / 0.05)
            u = torch.exp(-(((x - 0.5 * L) / W) ** 2)) + gate * net(
                torch.stack([x, t], -1)).squeeze(-1)
            u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
            u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
            c = corr(torch.stack([u, u_x], -1)).squeeze(-1)
            res = u_t - torch.exp(log_a) * u_xx - c
            (res.pow(2).mean() + 1e-3 * c.pow(2).mean()
             + 0.03 * u_xxx.pow(2).mean() * 1e-6).backward()
            opt.step()

    run(10)                                   # warm up kernels / autotune
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    run(epochs)
    if dev.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / epochs * 1e3     # ms/epoch


print(f"{'collocation':>12} {'cpu f64':>10} {'cpu f32':>10} {'mps f32':>10}"
      f" {'mps vs cpu f64':>15}")
cpu = torch.device("cpu")
mps = torch.device("mps")
for n in (2_000, 10_000, 50_000):
    a = step_time(cpu, torch.float64, n)
    b = step_time(cpu, torch.float32, n)
    c = step_time(mps, torch.float32, n)
    print(f"{n:>12,} {a:>9.1f}ms {b:>9.1f}ms {c:>9.1f}ms {a / c:>14.2f}x")
