"""The wave-function PINN: physics as the only supervision.

There is no data in these tests. The network is given a potential and a
boundary, and is checked against eigenvalues known in closed form and against
the same problem solved by direct diagonalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.methods.eigen_pinn import (  # noqa: E402
    COLLAPSE_TOL,
    WaveFunction,
    fit_eigen_pinn,
    fit_spectrum,
)
from physprior.problems.quantum import schrodinger as S  # noqa: E402


def _free(x):
    return np.zeros_like(x)


def _sho(x):
    return 0.5 * x**2


# --- the hard constraint --------------------------------------------------
def test_boundary_conditions_hold_exactly_before_any_training():
    """psi = (x-a)(b-x) * NN(x) satisfies psi(a) = psi(b) = 0 by
    construction, so there is no boundary penalty and no weight to tune."""
    net = WaveFunction(0.0, 1.0)
    ends = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert torch.allclose(net(ends), torch.zeros(2, dtype=torch.float64), atol=1e-14)


def test_the_boundary_stays_exact_after_training():
    st = fit_eigen_pinn(_free, 0.0, 1.0, epochs=200, seed=0)
    assert abs(st.psi[0]) < 1e-12
    assert abs(st.psi[-1]) < 1e-12


# --- the physics ----------------------------------------------------------
@pytest.mark.slow
def test_ground_state_of_the_infinite_well():
    """E_1 = pi^2 / 2, to three decimal places, from the physics alone."""
    st = fit_eigen_pinn(_free, 0.0, 1.0, epochs=2500, seed=11)
    assert st.energy == pytest.approx(np.pi**2 / 2, rel=2e-3)
    assert st.residual_rms < 0.5


@pytest.mark.slow
def test_ground_state_of_the_harmonic_oscillator():
    """E_0 = 1/2 omega. Equally spaced levels are the signature."""
    st = fit_eigen_pinn(_sho, -6.0, 6.0, epochs=2500, seed=11)
    assert st.energy == pytest.approx(0.5, rel=5e-3)


@pytest.mark.slow
def test_the_learned_spectrum_matches_the_diagonalised_one():
    """Against `solve_1d`, which is milliseconds and more accurate.

    The PINN is not supposed to win. It is supposed to agree, on a problem
    where an independent method can say what the answer is.
    """
    states = fit_spectrum(_free, 0.0, 1.0, n_levels=3, epochs=2500, seed=11)
    reference = S.infinite_well(length=1.0, n_levels=3)
    for st, exact in zip(states, reference.energy, strict=True):
        assert st.converged, f"level {st.level} collapsed onto a lower state"
        assert st.energy == pytest.approx(exact, rel=5e-3)


@pytest.mark.slow
def test_excited_states_come_out_orthogonal():
    """The orthogonality penalty is what selects an excited state at all."""
    states = fit_spectrum(_free, 0.0, 1.0, n_levels=3, epochs=2500, seed=11)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            assert states[i].overlap(states[j]) < COLLAPSE_TOL


@pytest.mark.slow
def test_a_collapsed_level_is_marked_rather_than_returned_silently():
    """Spectral bias is real: high levels sometimes come back as a copy of a
    lower one. The project's rule is that such a result is marked, not
    reported as if it converged."""
    states = fit_spectrum(_free, 0.0, 1.0, n_levels=3, epochs=250, retries=0, seed=3)
    for st in states:
        # whatever happened, the flag must agree with the measured overlap
        assert st.converged == (st.max_overlap <= COLLAPSE_TOL)


# --- the two formulations -------------------------------------------------
@pytest.mark.slow
def test_the_residual_formulation_also_finds_the_ground_state():
    """`residual` makes E a trainable parameter and divides by the norm, so
    the zero function cannot win by shrinking the loss."""
    st = fit_eigen_pinn(_free, 0.0, 1.0, method="residual", epochs=2500, seed=11)
    assert st.energy == pytest.approx(np.pi**2 / 2, rel=5e-2)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown method"):
        fit_eigen_pinn(_free, 0.0, 1.0, method="magic", epochs=1)


# --- the trivial solution -------------------------------------------------
@pytest.mark.slow
def test_the_learned_state_is_not_the_zero_function():
    """psi = 0 solves the equation exactly. Both objectives are ratios of
    inner products precisely so that it cannot be the minimum."""
    st = fit_eigen_pinn(_free, 0.0, 1.0, epochs=1500, seed=5)
    assert np.max(np.abs(st.psi)) > 1e-6
    assert np.trapezoid(st.normalised() ** 2, st.x) == pytest.approx(1.0, rel=1e-6)
