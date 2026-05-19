from abc import ABC, abstractmethod
from typing import override

import jaxtyping
import numpy as np
from common.lp_problem import LpProblem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex_util import get_non_basic_vars

PIVOTING_TOLERANCE = 1e-5


class PrimalPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        """
        Selects the variable that should enter the basis.

        Args:
            reduced_costs: reduced costs
            non_basic_vars: variable indices of the non basic variables, assumed to be sorted
        Returns:
            The variable index, i.e. an element of the array `non_basic_vars`, of a variable that should enter the basis.
        """
        ...

    @abstractmethod
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        Selects the index exiting the basis.

        If we compare the arguments with "Numerical Optimization", Nocedal & Wright, page 370,
        we can make the following identifications:

        `x_basis[p]` is the value for the decision variable `x_k`, where `k=basis[p]`.

        `basic_direction` is `d`, with `d = B^-1 * A_q`, where B is the basis matrix `A[:, basis]`,
        and `A_q = A[:, q]` is the column of the constraint matrix for the entering variable `x_q`.

        Args:
            basis: Variable indices for the basic variables.
            x_basis: Values for the basic variables.
            basic_direction: The basic direction for the entering variable.

        Returns:
            Index `p` in ``basis`` array for the decision variable that should be removed from the basis.
        """
        ...


class DualPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        TODO(martins): Describe purpose of picking entering index
        """
        ...

    @abstractmethod
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        s: jaxtyping.Float[ArrayF, " num_nonbasic"],
        pivot_direction: jaxtyping.Float[ArrayF, " num_nonbasic"],
    ) -> int:
        """
        TODO(martins): Describe purpose of picking exiting index
        """
        ...


def index_of_smallest_ratio(
    basis: jaxtyping.Int[ArrayI, " m"],
    x_basis: jaxtyping.Float[ArrayF, " m"],
    basic_direction: jaxtyping.Float[ArrayF, " m"],
) -> int:
    """
    Args:
        basis: Variable indices for the basic variables.
        x_basis: Values for the basic variables.
        basic_direction: The basic direction for the entering variable.

    Returns:
        Index `i` in `basis` array with the smallest positive ratio `x_basis[i] / basic_direction[i]`,
        choosing the index corresponding to the lowest variable index `basis[i]` in case of ties.
    """

    smallest_ratio_with_smallest_var_index = min(
        (max(0.0, float(x_basis[i])) / basic_direction[i], basis[i], i)
        for i in range(len(x_basis))
        if basic_direction[i] > (PIVOTING_TOLERANCE)
    )

    return smallest_ratio_with_smallest_var_index[2]


class BlandsRule(PrimalPivotingStrategy):
    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        candidate_indices = np.flatnonzero(reduced_costs < -PIVOTING_TOLERANCE)

        if len(candidate_indices) == 0:
            raise RuntimeError("No valid entering variable found.")

        return int(candidate_indices[np.argmin(non_basic_vars[candidate_indices])])

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return index_of_smallest_ratio(basis, x_basis, basic_direction)


class DantzigsRule(PrimalPivotingStrategy):
    """Dantzig's rule is one of the simplest pivoting strategies. It was
    suggested by George Dantzig, inventor of the Primal Simplex algorithm.
    It simply selectes the variable with the most negative reduced cost.

    See section 13.5 in "Numerical Optimization" for more details.

    Since all rules use smallest subscript for the exiting index, that is not tested
    """

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # non_basic_vars are sorted in my implementation
        return int(np.argmin(reduced_costs))

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return index_of_smallest_ratio(basis, x_basis, basic_direction)


class SteepestEdgeRule(PrimalPivotingStrategy):
    """Primal steepest-edge pricing for the entering variable."""

    def __init__(
        self,
        problem: LpProblem | None = None,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> None:
        self.problem: LpProblem | None = None
        self.entering_index = -1
        self.non_basic_vars = np.array([], dtype=int)
        self.norm_eta_squared = np.array([], dtype=float)

        if problem is not None and initial_basis is not None:
            self.initialize(problem, initial_basis)

    @override
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        self.problem = problem
        self.entering_index = -1
        self.non_basic_vars = get_non_basic_vars(problem.num_variables, basis)

        b_inv = np.linalg.inv(problem.constraint_matrix[:, basis])
        basic_directions = b_inv @ problem.constraint_matrix[:, self.non_basic_vars]

        self.norm_eta_squared = 1.0 + np.sum(
            basic_directions * basic_directions,
            axis=0,
        )

    def _update_eta(
        self,
        exiting_index: int,
        basis: jaxtyping.Int[ArrayI, " m"],
        b_inv: jaxtyping.Float[ArrayF, "m m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> None:
        if self.problem is None:
            raise RuntimeError("SteepestEdgeRule must be initialized before use.")

        entering_index = self.entering_index

        if entering_index < 0 or entering_index >= len(self.non_basic_vars):
            raise RuntimeError("Entering index is invalid.")

        non_basic_vars = self.non_basic_vars
        gamma = self.norm_eta_squared

        exiting_variable = int(basis[exiting_index])
        pivot = float(basic_direction[exiting_index])

        if abs(pivot) <= PIVOTING_TOLERANCE:
            raise RuntimeError("Pivot is too close to zero.")

        entering_gamma = float(gamma[entering_index])

        # Compute all old non-basic directions needed for the recurrence,
        # but avoid forming the full B_inv @ A_N matrix.
        non_basic_columns = self.problem.constraint_matrix[:, non_basic_vars]

        alpha = (b_inv[exiting_index, :] @ non_basic_columns) / pivot
        direction_dot_products = (basic_direction @ b_inv) @ non_basic_columns

        # Steepest-edge recurrence, applied in the current non-basic ordering.
        gamma -= 2.0 * alpha * direction_dot_products
        gamma += alpha * alpha * entering_gamma

        # The entering variable becomes basic. The exiting variable becomes
        # non-basic and takes the same slot in non_basic_vars.
        non_basic_vars[entering_index] = exiting_variable
        gamma[entering_index] = entering_gamma / (pivot * pivot)

        np.maximum(gamma, PIVOTING_TOLERANCE, out=gamma)

        self.entering_index = -1

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        if not np.array_equal(non_basic_vars, self.non_basic_vars):
            raise RuntimeError(
                "Steepest-edge weights are not aligned with non-basic variables."
            )

        candidate_indices = np.flatnonzero(reduced_costs < -PIVOTING_TOLERANCE)

        if len(candidate_indices) == 0:
            raise RuntimeError("No valid entering variable found.")

        scores = reduced_costs[candidate_indices] / np.sqrt(
            self.norm_eta_squared[candidate_indices]
        )

        self.entering_index = int(candidate_indices[np.argmin(scores)])
        return self.entering_index

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        if inv_basis_matrix is None:
            raise ValueError(
                "SteepestEdgeRule requires the current inverse basis matrix."
            )

        exiting_index = index_of_smallest_ratio(basis, x_basis, basic_direction)
        self._update_eta(
            exiting_index=exiting_index,
            basis=basis,
            b_inv=inv_basis_matrix,
            basic_direction=basic_direction,
        )

        return exiting_index

class DualBlandsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        negative_basic_vars = [
            (variable_index, basis_index, var)
            for (basis_index, var), variable_index in zip(
                enumerate(primal_vars), basic_vars, strict=True
            )
            if var < -PIVOTING_TOLERANCE
        ]
        return min(negative_basic_vars)[1]

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)


class DualDantzigsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return int(np.argmin(primal_vars))

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)


class DualSteepestEdgeRule(DualPivotingStrategy):
    """Dual steepest-edge leaving-row rule."""

    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        del basic_vars
        if inv_basis_matrix is None:
            raise ValueError(
                "DualSteepestEdgeRule requires the current inverse basis matrix."
            )

        row_norms_squared = np.sum(inv_basis_matrix * inv_basis_matrix, axis=1)
        candidate_mask = primal_vars < -PIVOTING_TOLERANCE
        candidate_scores = np.full_like(primal_vars, np.inf, dtype=float)
        # Exact dual steepest edge uses ||e_i^T B^-1|| as the edge length for
        # each candidate leaving row. Recomputing keeps the rule aligned with
        # the current basis after inverse refreshes and Phase I augmentation.
        candidate_scores[candidate_mask] = primal_vars[candidate_mask] / np.sqrt(
            np.maximum(row_norms_squared[candidate_mask], PIVOTING_TOLERANCE)
        )
        return int(np.argmin(candidate_scores))

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)
