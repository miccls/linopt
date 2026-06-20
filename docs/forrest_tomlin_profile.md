# Forrest-Tomlin factorization profile

Date: 2026-06-20

## Scope

The Simplex basis factorization on `agent-branch` represents the basis as
`B = P L R_1 ... R_k U_bar`:

- `L` is stored as a sparse unit lower triangular factor;
- each update appends a sparse Forrest-Tomlin row eta `R_i`;
- `U_bar` is stored as a sparse right factor with a cached sparse solve
  factorization for FTRAN, BTRAN, and update row solves.

The algebra follows the row-eta update in Huangfu/Hall equations 11-13. This is
still a correctness-first implementation: `U_bar` is rebuilt as a sparse matrix
after every update, and its sparse solve factorization is rebuilt after every
update. A production implementation should eventually keep the permuted
triangular right factor and bump rows in a more incremental form.

## Baseline

Environment: macOS, Python 3.14.4 via `uv`, branch `agent-branch`.

Full existing suite:

```sh
python3 - <<'PY'
import subprocess
try:
    raise SystemExit(subprocess.run(['uv','run','pytest'], timeout=180).returncode)
except subprocess.TimeoutExpired:
    print('TIMEOUT after 180s')
    raise SystemExit(124)
PY
```

Result: timed out after 180s. Before timeout, two pre-existing IPM failures were
observed:

- `common/tests/test_netlib_problems.py::test_netlib_ipm[pilotnov]`
- `common/tests/test_netlib_problems.py::test_netlib_ipm[dfl001]`

The Simplex `scsd8` case passed before the timeout.

Isolated Simplex `scsd8` baseline:

```sh
uv run pytest common/tests/test_netlib_problems.py::test_netlib_primal_simplex[scsd8] -q
```

Result: passed in 3.41s pytest time, 3.616s wrapper wall time.

## After Change

Focused Forrest-Tomlin tests:

```sh
uv run pytest simplex/tests/test_simplex.py::TestForrestTomlin -q
```

Result: 5 passed in 0.09s, 0.297s wrapper wall time.

Simplex tests:

```sh
uv run pytest simplex/tests/test_simplex.py -q
```

Result: 40 passed in 0.31s, 0.457s wrapper wall time.

Isolated Simplex `scsd8`:

```sh
uv run pytest common/tests/test_netlib_problems.py::test_netlib_primal_simplex[scsd8] -q
```

Result: passed in 5.22s pytest time, 5.427s wrapper wall time.

Branch-tip `scsd8` cProfile comparison:

| Version | Objective | Simplex iterations | cProfile wall time |
| --- | ---: | ---: | ---: |
| Baseline `215f98b` product-form column eta checkpoint | 904.999999925 | 802 | 2.695s |
| Branch tip `0d3262d` Forrest-Tomlin row eta update | 904.999999925 | 921 | 6.199s |

Top cumulative entries for the branch-tip cProfile run:

```text
4254 calls   3.894s  linear_algebra.py:update
4342 calls   1.022s  scipy.sparse.linalg.splu
12999 calls  1.174s  scipy.sparse.linalg.spsolve_triangular
4403 calls   0.748s  linear_algebra.py:btran
4342 calls   0.624s  linear_algebra.py:ftran
```

Full existing suite rerun:

```sh
python3 - <<'PY'
import subprocess
try:
    raise SystemExit(subprocess.run(['uv','run','pytest'], timeout=180).returncode)
except subprocess.TimeoutExpired:
    print('TIMEOUT after 180s')
    raise SystemExit(124)
PY
```

Result: ended with signal 143 before emitting a clean pytest summary or timeout
message. Before termination it had passed through `test_netlib_ipm[scsd8]`.

Final checks before PR update:

```sh
uv run pytest simplex/tests -q
uv run pytest 'common/tests/test_netlib_problems.py::test_netlib_primal_simplex[scsd8]' simplex/tests/test_simplex.py -q
uv run ruff check simplex/src/simplex/linear_algebra.py simplex/tests/test_simplex.py docs/forrest_tomlin_profile.md simplex/forrest_tomlin_update_notes.md
```

Results:

- `simplex/tests`: 46 passed in 0.39s.
- `scsd8` plus `simplex/tests/test_simplex.py`: 40 passed in 6.25s.
- Ruff: passed.

## Notes

The post-change `scsd8` run is slower than the recorded baseline. The tradeoff is
intentional for this step: the right-factor solve is explicit and cached, and the
update path uses the same sparse right-factor solve for `U_bar.T`. The next
performance step is to replace the per-update sparse LU rebuild of `U_bar` with
a true incremental permuted-triangular/bump solve.
