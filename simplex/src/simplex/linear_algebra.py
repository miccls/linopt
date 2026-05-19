import jaxtyping
import numpy as np
from common.numpy_type_aliases import ArrayF


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
