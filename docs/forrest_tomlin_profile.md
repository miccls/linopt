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

## Notes

The post-change `scsd8` run is slower than the recorded baseline. The tradeoff is
intentional for this step: the right-factor solve is explicit and cached, and the
update path uses the same sparse right-factor solve for `U_bar.T`. The next
performance step is to replace the per-update sparse LU rebuild of `U_bar` with
a true incremental permuted-triangular/bump solve.
