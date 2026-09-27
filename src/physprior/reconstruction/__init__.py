"""Reconstructing physical fields from sparse sensors, in 1, 2 and 3 dimensions.

`fields`       fields with known truth (box Poisson, free-space potentials)
`methods`      physics fit, PINN, GP, physics+GP, interpolation, MLP
`convergence`  the solvers' convergence studies
`study`        the sweeps: `run(quick)`
`plots`, `doc` figures and docs/reconstruction/README.md from results
"""
