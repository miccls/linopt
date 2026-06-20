# Forrest-Tomlin Basis Update Profile

Date: 2026-06-20

This note records the `scsd8` Netlib profile for the Simplex basis-factor update
change on `agent-branch`.

## Command

The before run used a detached worktree at commit `215f98b`. The after run used
the working tree after replacing the product-form column eta stack with explicit
Forrest-Tomlin row eta factors.

```bash
uv run python - <<'PY'
import cProfile
import io
import pstats
import time
from common import lp_problem
from common.netlib import load_netlib_problems
from simplex import pivoting_strategy, primal_simplex

name = "scsd8"
res = load_netlib_problems.download_and_parse_mps(name)
assert res is not None
A, b, c, row_types, _, _ = res
A_std, b_std, c_std = load_netlib_problems.convert_to_standard_form(
    A, b, c, row_types
)
problem = lp_problem.LpProblem(A_std, b_std, c_std)
solver = primal_simplex.PrimalSimplex(
    pivot_strategy=pivoting_strategy.DantzigsRule()
)
profiler = cProfile.Profile()
start = time.perf_counter()
profiler.enable()
solution = solver.solve(problem, max_iterations=100000)
profiler.disable()
elapsed = time.perf_counter() - start
objective = float(problem.objective.T @ solution.solution)
stats_io = io.StringIO()
pstats.Stats(profiler, stream=stats_io).strip_dirs().sort_stats(
    "cumtime"
).print_stats(15)
print(f"objective={objective:.12g}")
print(f"iterations={len(solver.history.objective_history) - 1}")
print(f"elapsed_seconds={elapsed:.6f}")
print(stats_io.getvalue())
PY
```

## Results

| Version | Objective | Simplex iterations | cProfile wall time |
| --- | ---: | ---: | ---: |
| Baseline `215f98b` product-form column eta checkpoint | 904.999999925 | 802 | 2.695 s |
| `b1e35fe` Forrest-Tomlin row eta branch tip | 904.999999925 | 921 | 6.199 s |

## Top Profile Entries

Baseline:

```text
10354 calls  1.900 s  scipy.sparse.linalg.spsolve_triangular
 3416 calls  1.018 s  linear_algebra.py:_u_ftran
 1761 calls  0.616 s  linear_algebra.py:_u_btran
 1690 calls  0.742 s  linear_algebra.py:update
```

After:

```text
4254 calls  3.894 s  linear_algebra.py:update
4342 calls  1.022 s  scipy.sparse.linalg.splu
12999 calls 1.174 s  scipy.sparse.linalg.spsolve_triangular
4403 calls  0.748 s  linear_algebra.py:btran
4342 calls  0.624 s  linear_algebra.py:ftran
```

## Notes

The new implementation represents the updated basis as `P L R_1 ... R_k U_bar`
and stores each Forrest-Tomlin row eta factor explicitly. `ftran` and `btran`
now apply the factored row operations rather than using a column eta
product-form update.

The remaining performance limitation is `U_bar`: after row updates it is a
row-eta-permuted triangular factor, but this first pass rebuilds a sparse LU of
that right factor on each update. The next optimization step should add a
dedicated permuted-triangular solve and avoid repeated `splu` factorizations
during `update`.
