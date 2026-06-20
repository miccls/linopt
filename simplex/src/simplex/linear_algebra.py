from dataclasses import dataclass
from typing import Any

import jaxtyping
import numpy as np
import scipy.linalg
from common.numpy_type_aliases import ArrayF, ArrayI
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve_triangular


@dataclass
class _RowEta:
    index: int
    indices: jaxtyping.Int[ArrayI, " nnz"]
    values: jaxtyping.Float[ArrayF, " nnz"]

    @classmethod
    def from_dense_row(
        cls,
        index: int,
        row: jaxtyping.Float[ArrayF, " m"],
    ) -> "_RowEta":
        row = np.asarray(row, dtype=np.float64).copy()
        row[index] = 0.0
        row[np.isclose(row, 0.0, atol=1e-14)] = 0.0
        nonzero_indices = np.flatnonzero(row)
        return cls(
            index=index,
            indices=nonzero_indices,
            values=np.asarray(row[nonzero_indices], dtype=np.float64),
        )

    def apply_inverse(
        self, vector: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve R x = vector for a unit row eta matrix R = I + e_p r.T."""
        result = vector.copy()
        result[self.index] -= float(self.values @ vector[self.indices])
        return result

    def apply_inverse_transpose(
        self, vector: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve R.T x = vector for a unit row eta matrix R = I + e_p r.T."""
        result = vector.copy()
        result[self.indices] -= self.values * vector[self.index]
        return result

    def matrix(self, matrix_size: int) -> sparse.csr_array:
        eta_matrix = sparse.eye(matrix_size, format="lil", dtype=np.float64)
        eta_matrix[self.index, self.indices] = sparse.csr_array(
            self.values.reshape(1, -1),
            shape=(1, len(self.indices)),
        )
        return eta_matrix.tocsr()


class _SparseRightFactor:
    """Cached sparse factorization for the current Forrest-Tomlin right factor."""

    def __init__(self, matrix: sparse.spmatrix | sparse.sparray) -> None:
        self.matrix = sparse.csc_array(matrix)
        self._factor: Any = splu(sparse.csc_matrix(self.matrix))

    def solve(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        return np.asarray(self._factor.solve(rhs), dtype=np.float64)

    def solve_transpose(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        return np.asarray(self._factor.solve(rhs, trans="T"), dtype=np.float64)


class ForrestTomlinFactorization:
    """
    Sparse Forrest-Tomlin basis factorization with factored row updates.

    The representation after k updates is
        B = P L R_1 R_2 ... R_k U_bar,
    so solving with the basis uses
        B^-1 = U_bar^-1 R_k^-1 ... R_2^-1 R_1^-1 L^-1 P^T.

    L is kept as a sparse triangular factor. Each pivot updates U_bar by first
    replacing the leaving basis column with the left-factor transformed entering
    column, then appending the Forrest-Tomlin row eta that restores the row
    structure described in Hall/Huang equations 11-13. U_bar is stored sparsely
    and solved as the current row-eta-permuted triangular factor.
    """

    def __init__(self, basis_matrix: jaxtyping.Float[ArrayF, "m m"]) -> None:
        permutation, l_factor, u_factor = scipy.linalg.lu(
            np.asarray(basis_matrix, dtype=np.float64)
        )
        self.permutation = sparse.csr_array(permutation)
        self.l_factor = sparse.csc_array(l_factor)
        self._right_factor = _SparseRightFactor(sparse.csc_array(u_factor))
        self.row_etas: list[_RowEta] = []

    @classmethod
    def from_factors(
        cls,
        l_factor: jaxtyping.Float[ArrayF, "m m"],
        u_factor: jaxtyping.Float[ArrayF, "m m"],
    ) -> "ForrestTomlinFactorization":
        factorization = cls.__new__(cls)
        factorization.permutation = sparse.eye(l_factor.shape[0], format="csr")
        factorization.l_factor = sparse.csc_array(l_factor)
        factorization._right_factor = _SparseRightFactor(sparse.csc_array(u_factor))
        factorization.row_etas = []
        return factorization

    @property
    def u_factor(self) -> sparse.csc_array:
        return self._right_factor.matrix

    def ftran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve B x = rhs, returning x = B^-1 rhs."""
        partial = self._left_factor_inverse_times(rhs)
        for row_eta in self.row_etas:
            partial = row_eta.apply_inverse(partial)
        return self._u_ftran(partial)

    def btran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve B.T x = rhs, returning x = B^-T rhs."""
        result = self._right_factor.solve_transpose(rhs)
        for row_eta in reversed(self.row_etas):
            result = row_eta.apply_inverse_transpose(result)
        result = np.asarray(
            spsolve_triangular(
                self.l_factor.T,
                result,
                lower=False,
                unit_diagonal=True,
            ),
            dtype=np.float64,
        )
        return np.asarray(self.permutation @ result, dtype=np.float64)

    def update(
        self,
        entering_column: jaxtyping.Float[ArrayF, " m"],
        exiting_index: int,
    ) -> None:
        """
        Replace basis position `exiting_index` by `entering_column`.

        This performs equations 11-13 of the Forrest-Tomlin update:
        1. Transform a_q by the current left factors and row etas.
        2. Replace column p of U by the transformed spike column.
        3. Append the row eta R that eliminates the old row-p off-diagonals.
        4. Apply R^-1 to the spiked U to get the new U_bar.
        """
        spike_column = self._left_factor_inverse_times(entering_column)
        for row_eta in self.row_etas:
            spike_column = row_eta.apply_inverse(spike_column)

        unit_exiting = np.zeros(self.u_factor.shape[0])
        unit_exiting[exiting_index] = 1.0
        u_dense = self.u_factor.toarray()
        partial_btran = self._right_factor.solve_transpose(unit_exiting)
        pivot = float(u_dense[exiting_index, exiting_index])
        if np.isclose(pivot, 0.0):
            raise np.linalg.LinAlgError(
                "ForrestTomlinFactorization update produced a zero U pivot."
            )

        row_eta = _RowEta.from_dense_row(
            exiting_index,
            unit_exiting - pivot * partial_btran,
        )

        spiked_u = u_dense.copy()
        spiked_u[:, exiting_index] = spike_column
        spiked_u[exiting_index, :] -= row_eta.values @ spiked_u[row_eta.indices, :]

        self.row_etas.append(row_eta)
        self._right_factor = _SparseRightFactor(sparse.csc_array(spiked_u))

    def to_matrix(self) -> jaxtyping.Float[ArrayF, "m m"]:
        """Reconstruct the represented basis matrix. Intended for tests/debugging."""
        result = self.permutation @ self.l_factor
        for row_eta in self.row_etas:
            result = result @ row_eta.matrix(result.shape[0])
        result = result @ self.u_factor
        return np.asarray(result.toarray(), dtype=np.float64)

    def _left_factor_inverse_times(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        result: ArrayF = np.asarray(
            spsolve_triangular(
                self.l_factor,
                self.permutation.T @ rhs,
                lower=True,
                unit_diagonal=True,
            ),
            dtype=np.float64,
        )
        return result

    def _u_ftran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        return self._right_factor.solve(rhs)


def update_inverse(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Uses the Sherman-Morrison formula,
    ```
    (B + u * v^T)^-1 = B^-1 - (B^-1 * u * v^T * B^-1)/(1 + v^T * B^-1 * u),
    ```
    with
    ```
    u = -B[:, exiting_index] + A[:, entering_variable],
    ```
    and `v` a vector with zeros everywhere except a 1 at `exiting_index`.

    The product of the old inverse with the column vector u becomes,
    ```
    B^-1 * u = -v + B^-1 * A[:, entering_variable] = -v + d,
    ```
    where `d` is the basic direction vector.

    Approximate time consumption on Optdev's Dell machines: 3.5 ms

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.
    """

    # O(m^2)
    d = b_inv @ a[:, [entering_variable]]
    v = np.zeros((a.shape[0], 1))
    v[exiting_index] = 1

    # O(m^2)
    return b_inv - ((-v + d) @ b_inv[[exiting_index], :]) / np.float64(d[exiting_index])


def update_inverse_gaussian(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Follows the method on page 97 of "Introduction to Linear Optimization",
    The new basis matrix is
    ```
    B_new = [A_B(1), A_B(2), ... , A_l, ...  A_B(m).
    ```
    where `l` is the entering index.
    To find the inverse of this matrix, we compute the product between this matrix
    and the old inverse
    ```
    B^-1 * B_new = [e_1, ...  d_l, e_m]
    ```
    If we now perform a set of row operations, `Q`, to turn the `l`th
    into the `l`th unit vector, we obtain the identity matrix,
    ```
    Q * B^-1 * B_new = I,
    ```
    thus, `Q * B^-1 = B_new^-1`.

    For efficiency, it suffices to just compute `d_l` (basic direction of entering variable)
    to determine the necessary row operations.

    Approximate time consumption on Optdev's Dell machines:  4.0 ms

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.

    """
    # O(m^2)
    d = b_inv @ a[:, [entering_variable]]
    dl = d[exiting_index]
    rowl = b_inv[exiting_index]

    # O(m^2)
    return np.array(
        [
            row - (rowl * d[i] / dl) if i != exiting_index else row / dl
            for i, row in enumerate(b_inv)
        ]
    )
