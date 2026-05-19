import numpy as np
import pytest
from common import lp_problem
from ipm.predictor_corrector import PredictorCorrector
from linopt_native import PivotRule, solve_primal_simplex_dense
from simplex import pivoting_strategy
from simplex.dual_simplex import DualPivotRule, DualSimplex
from simplex.primal_simplex import PrimalPivotRule, PrimalSimplex


def test_public_primal_simplex_dispatches_to_native() -> None:
    a = np.array(
        [
            [1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([20.0, 20.0, 20.0])
    c = np.array([-10.0, -12.0, -12.0, 0.0, 0.0, 0.0])

    result = PrimalSimplex(pivot_rule=PrimalPivotRule.DANTZIG).solve(
        lp_problem.LpProblem(a, b, c),
        initial_basis=np.array([3, 4, 5]),
    )

    assert result.objective_value == pytest.approx(-136.0)
    assert result.solution == pytest.approx(np.array([4.0, 4.0, 4.0, 0.0, 0.0, 0.0]))


def test_native_primal_simplex_accepts_pivot_rule_enum() -> None:
    a = np.array(
        [
            [1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([20.0, 20.0, 20.0])
    c = np.array([-10.0, -12.0, -12.0, 0.0, 0.0, 0.0])

    result = solve_primal_simplex_dense(
        a,
        b,
        c,
        np.array([3, 4, 5], dtype=np.int32),
        100,
        PivotRule.Dantzig,
    )

    assert result["objective"] == pytest.approx(-136.0)


def test_public_dual_simplex_dispatches_to_native() -> None:
    a = np.array(
        [
            [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([20.0, 20.0, 20.0, 4.0])
    c = np.array([-10.0, -12.0, -12.0, 0.0, 0.0, 0.0, 0.0])

    result = DualSimplex(pivot_rule=DualPivotRule.DANTZIG).solve(
        lp_problem.LpProblem(a, b, c),
        initial_basis=np.array([0, 1, 2, 6]),
    )

    assert result.objective_value == pytest.approx(-40.0)


def test_public_predictor_corrector_dispatches_to_native() -> None:
    a = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0, 0.0, 0.0])

    result = PredictorCorrector(100, 1e-10).solve(lp_problem.LpProblem(a, b, c))

    assert result.x == pytest.approx(np.array([0.0, 0.0, 1.0, 1.0]))


def test_public_primal_simplex_logs_match_python_shape(caplog: pytest.LogCaptureFixture) -> None:
    a = np.array(
        [
            [1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([20.0, 20.0, 20.0])
    c = np.array([-10.0, -12.0, -12.0, 0.0, 0.0, 0.0])

    with caplog.at_level("INFO", logger="simplex.primal_simplex"):
        PrimalSimplex(pivot_strategy=pivoting_strategy.DantzigsRule()).solve(
            lp_problem.LpProblem(a, b, c),
            initial_basis=np.array([3, 4, 5]),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Starting simplex algorithm..." in messages
    assert "Initial objective value 0" in messages
    assert "Iter     Objective      Primal Inf.    Dual Inf.    Time" in messages
    assert any(message.startswith("   1    ") for message in messages)
    assert any(
        message.startswith("Simplex algorithm found optimal objective -136 after")
        for message in messages
    )


def test_public_dual_simplex_logs_match_python_shape(caplog: pytest.LogCaptureFixture) -> None:
    a = np.array(
        [
            [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([20.0, 20.0, 20.0, 4.0])
    c = np.array([-10.0, -12.0, -12.0, 0.0, 0.0, 0.0, 0.0])

    with caplog.at_level("INFO", logger="simplex.dual_simplex"):
        DualSimplex(pivot_strategy=pivoting_strategy.DualDantzigsRule()).solve(
            lp_problem.LpProblem(a, b, c),
            initial_basis=np.array([0, 1, 2, 6]),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Starting Dual Simplex algorithm..." in messages
    assert "Initial objective value -136" in messages
    assert "Iter     Objective      Primal Inf.    Dual Inf.    Time" in messages
    assert any(message.startswith("   1    ") for message in messages)
    assert any(
        message.startswith("Simplex algorithm found optimal objective -40 after")
        for message in messages
    )


def test_public_predictor_corrector_logs_match_python_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    a = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 1.0, 0.0, 0.0])

    with caplog.at_level("INFO", logger="ipm.predictor_corrector"):
        PredictorCorrector(100, 1e-10).solve(lp_problem.LpProblem(a, b, c))

    messages = [record.getMessage() for record in caplog.records]
    assert "                Objective              Residual" in messages
    assert "Iter       Primal       Dual       Primal      Dual     Compl       Time" in messages
    assert any(message.startswith("   0   ") for message in messages)
