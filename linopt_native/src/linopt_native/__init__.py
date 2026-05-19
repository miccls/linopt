"""Private native acceleration package for linopt."""

from linopt_native._native import (  # noqa: F401
    LinoptNativeError,
    solve_dual_simplex_dense,
    solve_predictor_corrector_dense,
    solve_primal_simplex_dense,
)
