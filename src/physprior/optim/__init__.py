"""The optimization study: how physics-constrained models and black boxes train.

scratch            an MLP and five optimizers in plain NumPy (teaching code)
problems           the tasks (hydrogen, oscillator, heat), the three model
                   families, and Hessian / Lanczos curvature tools
optimizers_study   optimizers x learning rates x models x tasks
losses_study       loss functions under three noise models; the w_phys dial
pinn_balance       what the `balance` option does to the physics weight
report             run_all(), make_figures(), render_doc()
"""
