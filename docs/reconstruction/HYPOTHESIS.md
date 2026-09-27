# Field reconstruction: hypothesis, written before the study was run

Recorded 2026-09-27, before any sweep in `physprior.reconstruction.study` was
executed. The numbers that test it are in `results/reconstruction/` and on the
generated page `README.md` in this folder.

## Question

How does the value of a physics prior for reconstructing a physical field
from sparse point sensors change with the dimension of the field (1-D rod,
2-D plate, 3-D box)?

## Hypothesis (H-dim)

Let N*(d, method) be the number of noisy point sensors at which a method's
median field error (nRMSE on a dense grid, over the three reporting seeds)
first falls below a target of 0.10.

1. For the black-box and generic-smoothness methods (`nn`, `gp`, `interp`),
   N* grows roughly exponentially with d: log N* is close to linear in d,
   with a slope well above zero.
2. For the physics-constrained reconstruction (`physics`: the Green's
   function of the PDE in the loop, fitting source positions, strengths and
   boundary coefficients), N* grows with the number of physical unknowns P(d),
   which here grows linearly in d (P = K(d+1) + (d+1) for K sources in a box
   with a harmonic boundary lift). The ratio N*/P stays within a factor of
   about three across d.
3. Consequently the ratio N*(black box) / N*(physics) increases with d.

## What would refute it

- The best non-physics method reaches the target at 3-D with no more than
  three times the sensors it needs at 1-D (no exponential growth), or
- N*(physics) at 3-D exceeds three times the value predicted by scaling
  N*(physics, 1-D) with P(3)/P(1), or
- the ratio N*(best non-physics) / N*(physics) does not increase from 1-D to
  3-D.

Where a method never reaches the target inside the sensor budget, N* is
reported as a lower bound (censored), not extrapolated.

## Secondary questions, stated in advance

- Extrapolation: outside the convex hull of the sensors (box case) and
  outside the sensor cube (free-space case), the physics prior should keep
  its accuracy while the data-driven methods degrade.
- Model mismatch (2-D plate): with an unmodelled extra source or a wrong
  boundary condition, the physics fit is expected to stall at an error floor
  set by the missing physics; a Gaussian process on the physics residual
  (`pigp`) is expected to remove most of that floor once N is large enough
  to resolve the missing structure (the project's H3: a learned correction
  repairs an incomplete law).
- The PINN, which must find the sources from the PDE residual by gradient
  descent without the Green's function, is expected to sit between the
  physics fit and the black box.
