import re
from dataclasses import dataclass

import jaxtyping
import numpy as np
import pytest
from beartype import beartype as typechecker

# from typeguard import typechecked as typechecker # doesn't work at all with typeguard 4.4.4, the jaxtyping docs want you to use an old version 2.* as workaround..

# We use the explicit decorator @jaxtyping.jaxtyped here. In the simplex modules
# we use install_import_hook in the __init__.py to automatically add decorators to all functions.


@jaxtyping.jaxtyped(typechecker=typechecker)
def square_return_type_non_square_value(n: int) -> jaxtyping.Float[np.ndarray, "n n"]:
    return np.eye(n, n + 1)


def test_square_return_type_non_square_value() -> None:
    with pytest.raises(
        jaxtyping.TypeCheckError, match=re.escape("Actual value: f64[5,6](numpy)")
    ):
        square_return_type_non_square_value(5)


@jaxtyping.jaxtyped(typechecker=typechecker)
def fun_with_local_variable_annotations() -> float:
    """
    jaxtype only checks function arguments and return values. It will not error on the annotation of the local variable id_matrix
    """
    id_matrix: jaxtyping.Float[np.ndarray, "4 5"] = np.eye(5)
    assert id_matrix.shape == (5, 5)

    return float(id_matrix.trace())


@jaxtyping.jaxtyped(typechecker=typechecker)
def fun_with_arg_annotations(matrix: jaxtyping.Float[np.ndarray, "4 5"]) -> float:
    return float(matrix.trace())


def test_fun_with_internal_type_annotations() -> None:
    assert fun_with_local_variable_annotations() == 5.0
    with pytest.raises(jaxtyping.TypeCheckError):
        fun_with_arg_annotations(np.eye(5))


@jaxtyping.jaxtyped(typechecker=typechecker)
def produce_vector(
    vector: jaxtyping.Float[np.ndarray, " n"],
) -> jaxtyping.Float[np.ndarray, " n"]:
    longer_vector = np.zeros(len(vector) + 1)
    return longer_vector


def test_produce_vector() -> None:
    with pytest.raises(jaxtyping.TypeCheckError):
        produce_vector(np.zeros(2))


@jaxtyping.jaxtyped(typechecker=typechecker)
def produce_tuple(
    vector: jaxtyping.Float[np.ndarray, " n"],
) -> tuple[jaxtyping.Float[np.ndarray, "n n"], jaxtyping.Float[np.ndarray, " n"]]:
    longer_vector = np.zeros(len(vector) + 1)
    return np.outer(longer_vector, longer_vector), longer_vector


def test_checking_dims_in_output_tuple() -> None:
    result = produce_dataclass(np.zeros(2))  # no jaxtyping.TypeCheckError thrown here
    assert result.vector.shape == (3,)
    assert result.matrix.shape == (3, 3)


@jaxtyping.jaxtyped(typechecker=typechecker)
@dataclass
class ResultDataclass:
    matrix: jaxtyping.Float[np.ndarray, "n n"]
    vector: jaxtyping.Float[np.ndarray, " n"]


@jaxtyping.jaxtyped(typechecker=typechecker)
def produce_dataclass(vector: jaxtyping.Float[np.ndarray, " n"]) -> ResultDataclass:
    longer_vector = np.zeros(len(vector) + 1)
    # Result dimensions are not checked against " n" in input vector annotation,
    # only the scope in Result.__init__ is used.
    return ResultDataclass(
        matrix=np.outer(longer_vector, longer_vector), vector=longer_vector
    )


def test_checking_dims_in_output_dataclass() -> None:
    result = produce_dataclass(np.zeros(2))  # no jaxtyping.TypeCheckError thrown here
    assert result.vector.shape == (3,)
    assert result.matrix.shape == (3, 3)


def test_numpy_typing() -> None:
    f_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros((5, 5))
    f_vector: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(5)
    # f_matrix = f_vector  # mypy flags this!
    # assert f_matrix.shape == (5,)  # mypy flags this!, test passes

    g_vector = (
        f_matrix @ f_vector
    )  # shape information is lost in matmul, result type has shape type tuple[int,...]
    f_matrix = g_vector  # no error from mypy

    # i_vector: np.ndarray[tuple[int], np.dtype[np.int_]] = f_vector  # mypy flags this!
    # i_array: np.typing.NDArray[np.int_] = f_vector  # mypy flags this!
    f_array: np.typing.NDArray[np.float64] = (
        f_vector  # assigning to generic shape ok in mypy
    )
    f_vector = f_array  # assigning generic shape to fixed shape ok in mypy
