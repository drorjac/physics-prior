"""Expression trees, constant fitting and the Pareto-front selection rules."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from physprior.symbolic.expressions import (
    Candidate,
    OperatorSet,
    bi,
    complexity,
    const,
    evaluate,
    fit_constants,
    get,
    noise_floor,
    pareto_front,
    paths,
    put,
    scores,
    select,
    structure,
    to_string,
    un,
    var,
)

X = np.linspace(0.5, 3.0, 30)[:, None]


def planck_tree(c=1.0):
    return bi("/", un("cube", var()), bi("-", un("exp", var()), const(c)))


def test_evaluate_and_complexity():
    t = planck_tree()
    x = X[:, 0]
    np.testing.assert_allclose(evaluate(t, X), x**3 / np.expm1(x), rtol=1e-12)
    assert complexity(t) == 7


def test_string_round_trips_through_sympy_with_free_float_constants():
    t = planck_tree(1.0)
    e = sympy.sympify(to_string(t))
    # the constant must stay a Float, or a form check would treat it as fixed
    assert any(isinstance(a, sympy.Float) for a in e.atoms(sympy.Number))
    f = sympy.lambdify(sympy.Symbol("x0"), e, "numpy")
    np.testing.assert_allclose(f(X[:, 0]), evaluate(t, X), rtol=1e-12)


def test_invalid_arithmetic_is_nan_not_an_exception():
    t = un("log", bi("-", const(0.0), var()))
    assert np.all(np.isnan(evaluate(t, X)))


@pytest.mark.parametrize("method", ["lm", "bfgs"])
def test_fit_constants_recovers_the_constants(method):
    x = X[:, 0]
    y = 2.5 * np.exp(-0.7 * x)
    t = bi("*", const(1.0), un("exp", bi("*", const(-1.0), var())))
    fitted, loss = fit_constants(t, X, y, method=method, restarts=2)
    # BFGS on the scalar loss stops at a looser tolerance than LM
    assert loss < 1e-9
    np.testing.assert_allclose(evaluate(fitted, X), y, rtol=1e-4)


def test_put_and_paths_address_every_node():
    t = planck_tree()
    ps = list(paths(t))
    assert len(ps) == complexity(t)
    new = put(t, ps[-1], var())
    assert structure(get(new, ps[-1])) == "x0"


def test_operator_set_rejects_unknown_operators():
    with pytest.raises(ValueError):
        OperatorSet(("+",), ("tanh",))
    assert "exp" not in OperatorSet().without("exp").unary


def _c(k, loss):
    return Candidate(const(1.0), loss, k)


def test_front_drops_dominated_and_equal_loss_entries():
    front = pareto_front(
        [_c(1, 1.0), _c(3, 1.0 - 1e-15), _c(4, 0.1), _c(6, 0.2), _c(5, 0.01)]
    )
    assert [c.complexity for c in front] == [1, 4, 5]


def test_score_and_selection_match_pysr_definitions():
    """score_i = -log(l_i / l_{i-1}) / (c_i - c_{i-1}), first 0 (pysr
    calculate_scores); 'best' = max score among loss <= 1.5 min loss."""
    front = [_c(1, 1.0), _c(3, 1e-2), _c(5, 1e-6), _c(9, 0.9e-6)]
    s = scores(front)
    assert s[0] == 0.0
    assert s[1] == pytest.approx(np.log(100) / 2)
    assert s[2] == pytest.approx(np.log(1e4) / 2)
    assert select(front, "best").complexity == 5
    assert select(front, "accuracy").complexity == 9
    assert select(front, "score").complexity == 5


def test_best_rule_can_prefer_the_complex_end_when_losses_differ_by_more_than_1p5():
    front = [_c(1, 1.0), _c(3, 1e-4), _c(9, 1e-5)]
    assert select(front, "score").complexity == 3
    assert select(front, "best").complexity == 9


def test_noise_floor_scales_with_sigma_squared():
    assert noise_floor(0.1, 100) == pytest.approx(0.01 * (1 + 3 * np.sqrt(0.02)))
    assert noise_floor(0.0, 10) < 1e-11
