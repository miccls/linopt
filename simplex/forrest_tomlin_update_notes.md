# Forrest-Tomlin update notes

Date: 2026-06-20

## Scope

This change moves the Python simplex basis factorization from a fixed LU factor
with product-form column eta updates to a Forrest-Tomlin row-eta architecture:

```text
B_k = P L R_1 R_2 ... R_k U_k
B_k^-1 = U_k^-1 R_k^-1 ... R_1^-1 L^-1 P^T
```

The update follows Hall/Huang ERGO-13-001 equations 11-13:

1. Transform the entering column through `P`, `L`, and the existing row eta file.
2. Replace the leaving column of the current `U_k` with that transformed spike.
3. Compute the unit-pivot row eta `R` from the current U row.
4. Apply `R^-1` to the spiked U and append `R` to the factorization.

## Baseline and verification

All runs used bounded Python wrappers that stream output and terminate on timeout.

| Run | Result |
| --- | --- |
| Baseline `uv run --project simplex pytest simplex/tests -q` | 45 passed in 0.51s |
| Focused FT tests after change | 4 passed in 0.07s |
| Full simplex tests after change | 45 passed in 0.38s |
| Netlib primal simplex `afiro` after change | 1 passed in 0.29s |
| Netlib primal simplex `scsd8` after change | 1 passed in 3.59s |

## Current limitation

This is a correct incremental FT representation for the Python implementation,
not a full HiGHS-grade sparse update engine. The row eta file is factored and used
by FTRAN/BTRAN, but the current permuted upper factor `U_k` is still stored
explicitly and solved with SciPy sparse solves between periodic refactorizations.
A production sparse implementation should represent `U_k` itself as an eta-file
with deletion/insertion support for the leaving pivotal row/column.
