"""The Lorenz study: an inverse problem on a chaotic system.

Forty noisy observations of one Lorenz-63 trajectory, three Lyapunov times
long. The task is to reconstruct the state between the observations, recover
the three constants (sigma, rho, beta) of the law, and forecast past the end
of the window.

    system      the law, the reference integrator, the observations,
                the butterfly-effect ensemble and the Lyapunov exponent
    pinn        the network (black box or physics-informed) and its training
                recipe, with the derivative computed in forward mode
    classical   the fits that do not use a network: derivative regression,
                single and multiple shooting, and shooting started from a PINN
    metrics     state, derivative and constant errors, and the forecast horizon
    study       the experiments and their seeds
    speed       what each way of computing du/dt costs per training step
    figures     every figure, the block diagram and the butterfly animation
    doc         docs/lorenz/README.md, generated from results/lorenz

`physprior lorenz [--quick]` runs it end to end.
"""

TRACK = "lorenz"
