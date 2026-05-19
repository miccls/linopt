
# Research log

## Background

In this file I will keep notes on the state of the current solvers implemented and how the are performing on some netlib problems.
I currently have the following solvers:

* `PredictiorCorrector` : An interior point method
* `PrimalSimplex` : The Revised Simplex Algorithm
* `DualSimplex` : The Revised Dual Simplex Algorithm

## 29-03-2026

I think I may need to update the tolerances of the dual and primal methods as they reach the solution almost but then fail on some unexpected errors from numpy.
I have changed the production of `non_basic_variables` by looking if the index of a variable is in `basis` instead of `set(basis)`.
It could be beneficial to switch back at some point in the future, but for the problem sizes I have considering at the moment, it seems more efficient to use linear search.
Maybe basis should be a set from the get-go.

Example:
On the "stocfor2" I manage to overcome 41 iterations in 10s using:

```python
non_basic_vars = np.array(
    [i for i in range(problem.num_variables) if i not in basis]
)
```

but only 13 iterations in 10s using:

```python
non_basic_vars = np.array(
    [i for i in range(problem.num_variables) if i not in set(basis)]
)
```

## 05-04-2026

Benchmarking results for the current solver implementations on selected Netlib problems.  
*All tests were run with a **60s** hard timeout.*

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.02 |
| | Primal Simplex | 3 | 0.02 |
| | Dual Simplex | 11 | **0.01** |
| **adlittle** | IPM | 12 | **0.01** |
| | Primal Simplex | 97 | 0.39 |
| | Dual Simplex | 213 | 0.12 |
| **bandm** | IPM | 19 | **0.34** |
| | Primal Simplex | 1237 | 38.10 |
| | Dual Simplex | 3237 | 7.52 |
| **scsd1** | IPM | 8 | **0.10** |
| | Primal Simplex | 3168 | 13.31 |
| | Dual Simplex | 483 | 1.15 |
| **scsd6** | IPM | 10 | **0.51** |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | 5848 | 21.85 |
| **scsd8** | IPM | 10 | **0.82** |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |
| **stocfor2** | IPM | --- | *timeout* |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |

### Post-Optimization Benchmark (Basis Set Optimization)

After analyzing the `non_basic_vars` calculation, it was discovered that `set(basis)` inside a list comprehension causes Python to recreate the set object for every single iteration of the loop (once per variable).

Moving the set creation outside the loop:

```python
basis_set = set(basis)
non_basic_vars = np.array([i for i in range(n) if i not in basis_set])
```

yields a **10x speedup** on larger problems like `bandm` and `scsd1`.

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.02 |
| | Primal Simplex | 3 | 0.03 |
| | Dual Simplex | 11 | 0.03 |
| **adlittle** | IPM | 12 | 0.03 |
| | Primal Simplex | 97 | **0.13** |
| | Dual Simplex | 213 | **0.06** |
| **bandm** | IPM | 19 | 0.55 |
| | Primal Simplex | 1237 | **3.82** |
| | Dual Simplex | 3237 | **4.06** |
| **scsd1** | IPM | 8 | 0.12 |
| | Primal Simplex | 3168 | **1.33** |
| | Dual Simplex | 483 | **0.47** |
| **scsd6** | IPM | 10 | **0.25** |
| | Primal Simplex | --- | *error* |
| | Dual Simplex | 5848 | **7.08** |
| **scsd8** | IPM | 10 | **0.75** |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |
| **stocfor2** | IPM | --- | *timeout* |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |

### Observations

* The **10x speedup** in Simplex methods proves that the previous bottleneck was indeed the O(N*M) set recreation in the loop.

* **IPM** remains the most robust choice for scaling to larger problems, but Simplex is now much more competitive on the medium-sized Netlib sets.
* `scsd6` Primal Simplex still encounters an error, which likely warrants investigation into numerical stability.

## 06-04-2026

Due to suspicions that Bland's rule as a pivoting rule is not adequate for acceptable performance, I switched to Dantzig's rule to check the impact.
Not with the idea that this would solve the performance problem, but just to survey the impact.

Here are the results:

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.02 |
| | Primal Simplex | 3 | 0.02 |
| | Dual Simplex | 11 | 0.03 |
| **adlittle** | IPM | 12 | 0.03 |
| | Primal Simplex | 97 | **0.05** |
| | Dual Simplex | 213 | **0.06** |
| **bandm** | IPM | 19 | 0.55 |
| | Primal Simplex | 1237 | LinAlgError: Singular matrix |
| | Dual Simplex | 3237 | **4.06** |
| **scsd1** | IPM | 8 | 0.12 |
| | Primal Simplex | 3168 | **0.12** |
| | Dual Simplex | 483 | **0.47** |
| **scsd6** | IPM | 10 | **0.25** |
| | Primal Simplex | --- | 0.69 |
| | Dual Simplex | 5848 | **7.08** |
| **scsd8** | IPM | 10 | **0.75** |
| | Primal Simplex | --- | 16.31 |
| | Dual Simplex | --- | *timeout* |
| **stocfor2** | IPM | --- | *timeout* |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |

This was clearly an improvement, so I think it is worth to try some fancier pivoting rules next!

## 18-04-2026

New hardware baseline for the solver implementations on the selected Netlib problems in `test_netlib_problems.py`.
The simplex methods use the current Dantzig pivoting rules from the test file.
*All tests were run with a **60s** hard timeout.*

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.00 |
| | Primal Simplex | 0 | 0.00 |
| | Dual Simplex | 12 | 0.00 |
| **adlittle** | IPM | 12 | 0.01 |
| | Primal Simplex | 60 | 0.02 |
| | Dual Simplex | 77 | 0.01 |
| **bandm** | IPM | 19 | 0.04 |
| | Primal Simplex | --- | *LinAlgError: Singular matrix* |
| | Dual Simplex | 386 | 0.16 |
| **scsd1** | IPM | 8 | 0.01 |
| | Primal Simplex | 246 | 0.07 |
| | Dual Simplex | 209 | 0.06 |
| **scsd6** | IPM | 10 | 0.02 |
| | Primal Simplex | 458 | 0.20 |
| | Dual Simplex | --- | *LinAlgError: Singular matrix* |
| **scsd8** | IPM | 10 | 0.09 |
| | Primal Simplex | 749 | 2.61 |
| | Dual Simplex | --- | *LinAlgError: Singular matrix* |
| **stocfor2** | IPM | 22 | 4.33 |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | 1381 | 37.48 |

### Rerun with Dual Bland's Rule

After switching the dual simplex pivoting strategy back to `DualBlandsRule`, I reran the same benchmark.
The dual simplex singular matrix errors are fixed for `bandm` and `scsd6`, but `scsd1` still raises a singular matrix error.
*All tests were run with a **60s** hard timeout.*

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.00 |
| | Primal Simplex | 0 | 0.00 |
| | Dual Simplex | 7 | 0.00 |
| **adlittle** | IPM | 12 | 0.01 |
| | Primal Simplex | 60 | 0.02 |
| | Dual Simplex | 188 | 0.03 |
| **bandm** | IPM | 19 | 0.04 |
| | Primal Simplex | --- | *LinAlgError: Singular matrix* |
| | Dual Simplex | 3345 | 1.38 |
| **scsd1** | IPM | 8 | 0.01 |
| | Primal Simplex | 246 | 0.08 |
| | Dual Simplex | --- | *LinAlgError: Singular matrix* |
| **scsd6** | IPM | 10 | 0.02 |
| | Primal Simplex | 458 | 0.19 |
| | Dual Simplex | 5564 | 2.62 |
| **scsd8** | IPM | 10 | 0.09 |
| | Primal Simplex | 749 | 2.60 |
| | Dual Simplex | --- | *timeout* |
| **stocfor2** | IPM | 22 | 4.20 |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |

### Rerun after Disabling Runtime Type Checking

After disabling runtime type checking from `jaxtyping`, I reran the same benchmark.
The iteration counts and failures are unchanged from the previous `DualBlandsRule` run, but the simplex timings improve modestly.
*All tests were run with a **60s** hard timeout.*

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 9 | 0.01 |
| | Primal Simplex | 0 | 0.00 |
| | Dual Simplex | 7 | 0.00 |
| **adlittle** | IPM | 12 | 0.01 |
| | Primal Simplex | 60 | 0.02 |
| | Dual Simplex | 188 | 0.02 |
| **bandm** | IPM | 19 | 0.05 |
| | Primal Simplex | --- | *LinAlgError: Singular matrix* |
| | Dual Simplex | 3345 | 1.23 |
| **scsd1** | IPM | 8 | 0.01 |
| | Primal Simplex | 246 | 0.06 |
| | Dual Simplex | --- | *LinAlgError: Singular matrix* |
| **scsd6** | IPM | 10 | 0.02 |
| | Primal Simplex | 458 | 0.16 |
| | Dual Simplex | 5564 | 2.44 |
| **scsd8** | IPM | 10 | 0.10 |
| | Primal Simplex | 749 | 2.51 |
| | Dual Simplex | --- | *timeout* |
| **stocfor2** | IPM | 22 | 4.17 |
| | Primal Simplex | --- | *timeout* |
| | Dual Simplex | --- | *timeout* |

## 09-05-2026

After switching the IPM linear algebra to sparse matrices, I reran the Netlib benchmark for IPM only using `benchmark_netlib.py`.
The simplex implementations were not rerun in this pass.

| Problem | Solver | Iterations | Time (s) |
| :--- | :--- | :---: | :---: |
| **afiro** | IPM | 8 | 0.03 |
| **adlittle** | IPM | 11 | 0.02 |
| **bandm** | IPM | 17 | 0.05 |
| **scsd1** | IPM | 7 | 0.02 |
| **scsd6** | IPM | 10 | 0.02 |
| **scsd8** | IPM | 9 | 0.04 |
| **stocfor2** | IPM | 21 | 0.29 |

With the benchmark tolerance set to `5e-9`, `stocfor2` now solves successfully before the sparse normal-equation system becomes singular.
