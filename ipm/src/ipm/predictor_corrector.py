import logging

import numpy as np
from common import lp_problem
from linopt_native import solve_predictor_corrector_dense

from ipm import ipm_tools

logger = logging.getLogger(__name__)


def calculate_starting_point(
    problem: lp_problem.LpProblem,
) -> ipm_tools.PrimalDualTuple:
    """Calculates a suitable starting point for the predictor corrector algorithm
    according to the recipe on p. 410 in Nocedal & Wright"""
    aat_inv = np.linalg.inv(problem.constraint_matrix @ problem.constraint_matrix.T)

    x_tilde = problem.constraint_matrix.T @ aat_inv @ problem.rhs
    lam_tilde = aat_inv @ problem.constraint_matrix @ problem.objective
    s_tilde = problem.objective - problem.constraint_matrix.T @ lam_tilde

    delta_x = max(0.0, -1.5 * float(np.min(x_tilde)))
    delta_s = max(0.0, -1.5 * float(np.min(s_tilde)))

    x_hat = x_tilde + delta_x * np.ones(x_tilde.shape)
    s_hat = s_tilde + delta_s * np.ones(s_tilde.shape)

    delta_x_hat = 0.5 * (x_hat.T @ s_hat) / (sum(s_hat))
    delta_s_hat = 0.5 * (x_hat.T @ s_hat) / (sum(x_hat))

    x_0 = x_hat + delta_x_hat * np.ones(x_hat.shape)
    s_0 = s_hat + delta_s_hat * np.ones(s_hat.shape)
    lam_0 = lam_tilde
    return ipm_tools.PrimalDualTuple(x=x_0, lam=lam_0, s=s_0)


def update_point(
    point: ipm_tools.PrimalDualTuple,
    step: ipm_tools.PrimalDualTuple,
    primal_step_size: float,
    dual_step_size: float,
) -> ipm_tools.PrimalDualTuple:
    """Updates the current solution using a primal dual step and
    corresponding step sizes. Corresponds to the last two steps in the
    for-loop of algorithm 14.3 on p. 411 in the book."""
    return ipm_tools.PrimalDualTuple(
        x=point.x + primal_step_size * step.x,
        lam=point.lam + dual_step_size * step.lam,
        s=point.s + dual_step_size * step.s,
    )


class PredictorCorrector:
    def __init__(self, max_iterations: int, optimality_tolerance: float) -> None:
        self.max_iterations = max_iterations
        self.optimality_tolerance = optimality_tolerance

    def solve(self, problem: lp_problem.LpProblem) -> ipm_tools.PrimalDualTuple:
        """Solves the LP `problem` using algorithm 14.3 on p. 411 in Nocedal & Wright."""

        point = calculate_starting_point(problem)

        logger.info("                Objective              Residual")
        logger.info(
            "Iter       Primal       Dual       Primal      Dual     Compl       Time"
        )

        native_result = solve_predictor_corrector_dense(
            np.asarray(problem.constraint_matrix, dtype=float),
            np.asarray(problem.rhs, dtype=float),
            np.asarray(problem.objective, dtype=float),
            {"x": point.x, "lam": point.lam, "s": point.s},
            self.max_iterations,
            self.optimality_tolerance,
            logger.info,
        )
        final_point = ipm_tools.PrimalDualTuple(
            x=np.asarray(native_result["x"], dtype=float),
            lam=np.asarray(native_result["lam"], dtype=float),
            s=np.asarray(native_result["s"], dtype=float),
        )
        return final_point
