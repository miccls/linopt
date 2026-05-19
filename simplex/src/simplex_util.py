from dataclasses import dataclass

import jaxtyping
import numpy as np
from common.numpy_type_aliases import ArrayF, ArrayI


@dataclass(frozen=True)
class SolveResult:
    basis: jaxtyping.Int[ArrayI, " m"]
    solution: jaxtyping.Float[ArrayF, " n"]
    objective_value: float


class SolveFailedError(RuntimeError):
    pass


class InfeasibleLpError(SolveFailedError):
    pass


class UnboundedLpError(SolveFailedError):
    pass


class SimplexCyclingError(SolveFailedError):
    pass


class IterationLimitError(SolveFailedError):
    pass


MIN_CYCLE_LEN = 2
OBJECTIVE_IMPROVEMENT_TOL = 1e-9
OPTIMALITY_TOL = 1e-9
NON_NEGATIVITY_TOLERANCE = 1e-5  # Should not be smaller than pivoting tolerance...
INVERSE_RECOMPUTE_INTERVAL = 50


class SolveHistory:
    def __init__(self) -> None:
        self.basis_: list[tuple[int, ...]] = []
        self.objective_: list[float] = []
        self.basis_cycle_: set[tuple[int, ...]] = set()

    @property
    def basis_history(self) -> list[tuple[int, ...]]:
        return self.basis_

    @property
    def objective_history(self) -> list[float]:
        return self.objective_

    def update(
        self, basis: jaxtyping.Int[ArrayI, " m"], objective_value: float
    ) -> None:
        self.objective_.append(objective_value)

        if (
            len(self.objective_) > 1
            and abs(self.objective_[-1] - self.objective_[-2])
            > OBJECTIVE_IMPROVEMENT_TOL
        ):
            # If the objective improved, we are not cycling, so we can clear the history.
            self.basis_cycle_.clear()

        basis_signature = tuple(int(b) for b in sorted(basis))
        self.basis_.append(basis_signature)

        if basis_signature in self.basis_cycle_:
            raise SimplexCyclingError(
                f"Basis cycle of length {len(self.basis_cycle_)} detected"
            )
        self.basis_cycle_.add(basis_signature)


def get_non_basic_vars(
    num_variables: int, basis: jaxtyping.Int[ArrayI, "f{num_variables} - m"]
) -> jaxtyping.Int[ArrayI, " m"]:
    """
    Helper function to pick out the non basic variables.
    An important note for performance, if the set is computed inside
    the list comprehension as

        [i for i in range(num_variables) if i not in set(basis_set)]

    This has severe impacts on performance as python is not smart enough to not
    recompute the set every iteration.
    """
    basis_set = set(basis)
    return np.array([i for i in range(num_variables) if i not in basis_set])
