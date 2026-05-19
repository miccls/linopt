import logging

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI
from linopt_native import solve_primal_simplex_dense

from simplex import linear_algebra, pivoting_strategy
from simplex_util import (
    OPTIMALITY_TOL,
    InfeasibleLpError,
    IterationLimitError,
    SolveFailedError,
    SolveHistory,
    SolveResult,
    UnboundedLpError,
)

logger = logging.getLogger(__name__)


def _native_primal_pivot_rule(
    strategy: pivoting_strategy.PrimalPivotingStrategy,
) -> str:
    if isinstance(strategy, pivoting_strategy.BlandsRule):
        return "bland"
    if isinstance(strategy, pivoting_strategy.DantzigsRule):
        return "dantzig"
    if isinstance(strategy, pivoting_strategy.SteepestEdgeRule):
        return "steepest_edge"
    raise TypeError(
        "Native primal simplex only supports BlandsRule, DantzigsRule, and "
        "SteepestEdgeRule pivot strategies."
    )


def is_linearly_independent(
    problem: lp_problem.LpProblem,
    inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    entering_var: int,
    exiting_index: int,
) -> bool:
    pivot_column_entry = (
        inv_basis_matrix[exiting_index] @ problem.constraint_matrix[:, entering_var]
    )
    tolerance = 1e-9
    return bool(abs(pivot_column_entry) > tolerance)


def purge_aux_vars(
    problem: lp_problem.LpProblem,
    basis: jaxtyping.Int[ArrayI, " m"],
    num_variables: int,
) -> jaxtyping.Int[ArrayI, " m"]:
    aux_vars_still_in_basis = [b for b in basis if not b < num_variables]
    if aux_vars_still_in_basis:
        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])

        while aux_vars_still_in_basis:
            exiting_variable = aux_vars_still_in_basis.pop()
            exiting_index = next(
                i for i, x in enumerate(basis) if x == exiting_variable
            )

            # Generator of vars not in basis:
            non_basic_vars = (b for b in range(num_variables) if b not in basis)
            while not is_linearly_independent(
                problem,
                inv_basis_matrix,
                (entering_var := next(non_basic_vars)),
                exiting_index=exiting_index,
            ):
                pass

            basis[exiting_index] = entering_var

            inv_basis_matrix = linear_algebra.update_inverse(
                problem.constraint_matrix,
                inv_basis_matrix,
                entering_variable=entering_var,
                exiting_index=exiting_index,
            )
    return basis


class PrimalSimplex:
    pivoting_strategy_: pivoting_strategy.PrimalPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.PrimalPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.BlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def find_initial_basis(
        self, problem: lp_problem.LpProblem, max_iterations: int = 100
    ) -> jaxtyping.Int[ArrayI, " {problem.constraint_matrix.shape[0]}"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throws a SolveFailedError if it fails.

        Need to use the symbolic expression " {problem.constraint_matrix.shape[0]}" instead of " m" in the
        jaxtyping annotation. The constraint_matrix field in the LpProblem dataclass is annotated with "m n",
        but the runtime type checker can't see the annotations inside the dataclass, see
        https://github.com/patrick-kidger/jaxtyping/issues/342.
        """

        logger.info("Solving auxiliary phase one LP to find a starting basis...")

        e_matrix = np.diag(np.array([1.0 if b >= 0 else -1.0 for b in problem.rhs]))
        phase_one_problem = lp_problem.LpProblem(
            constraint_matrix=np.concatenate(
                [problem.constraint_matrix, e_matrix], axis=1
            ),
            rhs=np.array(problem.rhs),
            objective=np.concatenate(
                [np.zeros(problem.num_variables), np.ones(len(problem.rhs))]
            ),
        )
        # Use the smallest subscript rule to hopefully basis containing the original variables
        phase_one_solver = PrimalSimplex(pivot_strategy=self.pivoting_strategy_)
        phase_one_result = phase_one_solver.solve(
            phase_one_problem,
            initial_basis=np.array(
                range(problem.num_variables, phase_one_problem.num_variables)
            ),
            max_iterations=max_iterations,
        )
        if phase_one_result.objective_value > OPTIMALITY_TOL:
            raise SolveFailedError(
                f"Phase one objective value {phase_one_result.objective_value} is positive: Original problem is infeasible"
            )

        max_print_size = 10
        logger.info(
            f"Found starting basis {phase_one_result.basis if len(phase_one_result.basis) < max_print_size else ''}"
        )

        # Pivot out any auxiliary variables that may be in the basis
        basis = purge_aux_vars(
            phase_one_problem, phase_one_result.basis, problem.num_variables
        )

        aux_vars_still_in_basis = [b for b in basis if not b < problem.num_variables]
        if aux_vars_still_in_basis:
            raise SolveFailedError("Auxiliary variables still present in the basis.")

        return basis

    def _compute_reduced_costs(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    ) -> jaxtyping.Float[ArrayF, " num_nonbasic"]:
        lambda_t = inv_basis_matrix.T @ problem.objective[basis]
        return problem.objective[non_basic_vars] - problem.constraint_matrix[:, non_basic_vars].T @ lambda_t

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
    ) -> SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        return SolveResult(
            basis=basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> SolveResult:
        self.solve_history_ = SolveHistory()

        if initial_basis is not None:
            basis = np.array(initial_basis)
        else:
            try:
                basis = self.find_initial_basis(problem, max_iterations=max_iterations)
            except SolveFailedError as e:
                raise InfeasibleLpError(
                    f"Failed to find an initial simplex basis: {e}"
                ) from e
        logger.info("Starting simplex algorithm...")
        native_result = solve_primal_simplex_dense(
            np.asarray(problem.constraint_matrix, dtype=float),
            np.asarray(problem.rhs, dtype=float),
            np.asarray(problem.objective, dtype=float),
            np.asarray(basis, dtype=np.int32),
            max_iterations,
            _native_primal_pivot_rule(self.pivoting_strategy_),
            logger.info,
        )
        for native_basis, objective in zip(
            native_result["basis_history"],
            native_result["objective_history"],
            strict=True,
        ):
            self.solve_history_.update(np.asarray(native_basis, dtype=int), objective)

        if native_result["status"] == "unbounded":
            raise UnboundedLpError(native_result["message"])
        if native_result["status"] == "iteration_limit":
            raise IterationLimitError(native_result["message"])
        if native_result["status"] != "optimal":
            raise SolveFailedError(native_result["message"])

        return SolveResult(
            basis=np.asarray(native_result["basis"], dtype=int),
            solution=np.asarray(native_result["solution"], dtype=float),
            objective_value=float(native_result["objective"]),
        )
