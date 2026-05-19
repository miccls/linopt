import logging

import jaxtyping
import numpy as np
import scipy.linalg
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI
from linopt_native import solve_dual_simplex_dense

import simplex_util
from simplex import pivoting_strategy
from simplex_util import (
    NON_NEGATIVITY_TOLERANCE,
    SolveHistory,
    get_non_basic_vars,
)

logger = logging.getLogger(__name__)


def _native_dual_pivot_rule(strategy: pivoting_strategy.DualPivotingStrategy) -> str:
    if isinstance(strategy, pivoting_strategy.DualBlandsRule):
        return "dual_bland"
    if isinstance(strategy, pivoting_strategy.DualDantzigsRule):
        return "dual_dantzig"
    if isinstance(strategy, pivoting_strategy.DualSteepestEdgeRule):
        return "dual_steepest_edge"
    raise TypeError(
        "Native dual simplex only supports DualBlandsRule, DualDantzigsRule, "
        "and DualSteepestEdgeRule pivot strategies."
    )

# Solve with primal simplex
# N^T \lambda + s = c
# Add q
# N^T \lambda + s + Eq = c
# with E_ii = -1 if c<0 else 1
# Then solve min q subject to that constraint
# with initial basis all of the q's and q = |c|.


class DualSimplex:
    pivoting_strategy_: pivoting_strategy.DualPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.DualPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.DualBlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def _setup_artificial_problem(
        self, problem: lp_problem.LpProblem, big_m: float = 1e6
    ) -> tuple[
        lp_problem.LpProblem,
        jaxtyping.Int[ArrayI, " m"],
    ]:
        """
        Sets up the problem for a Phase 1 Dual Simplex by adding a single artificial constraint.

        This method handles problems where an initial basis is not dual-feasible by adding one
        artificial constraint and one artificial variable. This allows us to perform a "magic pivot"
        that immediately yields a dual-feasible basis.

        Mathematical Formulation:
        1. Start with an arbitrary basis B and compute its reduced costs s.
        2. If some s_k < 0, the basis is not dual-feasible.
        3. Augment the primal problem with a new constraint:
           \\sum_{j \\notin B} x_j + x_{art} = M
           where x_{art} is an artificial slack variable with cost 0, and M is a large scalar.
        4. The initial basis for this augmented problem is B U {x_{art}}.
        5. We force x_{art} to exit the basis and x_k (where k = argmin(s)) to enter. This is the "magic pivot".
           Because the artificial row has a coefficient of 1 for all non-basic variables, pivoting
           on it subtracts s_k from all reduced costs: s_j' = s_j - s_k.
           Since s_k is the minimum reduced cost, s_j' >= 0 for all j, making the new basis
           B U {x_k} strictly dual-feasible!

        For reference, this is commonly known as the "Single Artificial Constraint" method
        for the dual simplex method.
        """
        m, n = problem.constraint_matrix.shape

        # Use rank-revealing QR factorization to find m linearly independent columns.
        # The permutation array `p` pushes independent columns to the front.
        _, _, p = scipy.linalg.qr(problem.constraint_matrix, pivoting=True)
        basis = p[:m]

        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
        non_basic_vars = get_non_basic_vars(problem.num_variables, basis)

        lam = inv_basis_matrix.T @ problem.objective[basis]
        s_non_basic = problem.objective[non_basic_vars] - (
            problem.constraint_matrix[:, non_basic_vars].T @ lam
        )

        if np.all(s_non_basic >= -NON_NEGATIVITY_TOLERANCE):
            return problem, basis

        entering_index_in_non_basic = int(np.argmin(s_non_basic))
        entering_variable = non_basic_vars[entering_index_in_non_basic]

        # Augment the problem
        new_row = np.zeros(n + 1)
        new_row[non_basic_vars] = 1.0
        new_row[-1] = 1.0  # x_art

        new_a = np.zeros((m + 1, n + 1))
        new_a[:m, :n] = problem.constraint_matrix
        new_a[m, :] = new_row

        new_b = np.append(problem.rhs, big_m)
        new_c = np.append(problem.objective, 0.0)

        augmented_problem = lp_problem.LpProblem(
            constraint_matrix=new_a, rhs=new_b, objective=new_c
        )

        # The new basis is the original basis B U {entering_variable}
        new_basis = np.append(basis, entering_variable)
        return augmented_problem, new_basis

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        is_augmented: bool = False,
        original_num_variables: int = 0,
    ) -> simplex_util.SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        final_basis = basis

        if is_augmented:
            x_art_val = solution[-1]
            if x_art_val <= NON_NEGATIVITY_TOLERANCE:
                raise simplex_util.UnboundedLpError(
                    "Artificial bound reached, original Primal problem is unbounded (Dual is infeasible)."
                )

            art_index = problem.num_variables - 1
            final_basis = np.array([b for b in basis if b != art_index])
            solution = solution[:original_num_variables]

        return simplex_util.SolveResult(
            basis=final_basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> simplex_util.SolveResult:

        original_num_variables = problem.num_variables
        is_augmented = False

        if initial_basis is None:
            problem, initial_basis = self._setup_artificial_problem(problem)
            is_augmented = problem.num_variables > original_num_variables

        logger.info("Starting Dual Simplex algorithm...")
        native_result = solve_dual_simplex_dense(
            np.asarray(problem.constraint_matrix, dtype=float),
            np.asarray(problem.rhs, dtype=float),
            np.asarray(problem.objective, dtype=float),
            np.asarray(initial_basis, dtype=np.int32),
            max_iterations,
            _native_dual_pivot_rule(self.pivoting_strategy_),
            logger.info,
        )
        for native_basis, objective in zip(
            native_result["basis_history"],
            native_result["objective_history"],
            strict=True,
        ):
            self.solve_history_.update(np.asarray(native_basis, dtype=int), objective)

        if native_result["status"] == "infeasible":
            raise simplex_util.InfeasibleLpError(native_result["message"])
        if native_result["status"] == "iteration_limit":
            raise simplex_util.IterationLimitError(native_result["message"])
        if native_result["status"] != "optimal":
            raise simplex_util.SolveFailedError(native_result["message"])

        solution = np.asarray(native_result["solution"], dtype=float)
        basis = np.asarray(native_result["basis"], dtype=int)
        x_basis = solution[basis]
        return self._finalize_result(
            problem, basis, x_basis, is_augmented, original_num_variables
        )
