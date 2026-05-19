from dataclasses import dataclass
from functools import cached_property

from scipy import sparse

import jaxtyping

from common.numpy_type_aliases import ArrayF


@dataclass(frozen=True)
class LpProblem:
    constraint_matrix: jaxtyping.Float[ArrayF, "m n"]
    rhs: jaxtyping.Float[ArrayF, " m"]
    objective: jaxtyping.Float[ArrayF, " n"]

    @property
    def num_variables(self) -> int:
        return len(self.objective)

    @cached_property
    def sparse_constraint_matrix(self) -> sparse.csr_array:
        return sparse.csr_array(self.constraint_matrix)
