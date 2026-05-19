# Interior Point Method Implementation

## Introduction

Welcome to the IPM programming assignment! The goal of this assignment is to implement the Predictor Corrector Algorithm (Algorithm 14.3 in Nocedal & Wright) for solving linear programming problems on standard form:

Minimize `c^T * x`,
subject to `A * x = b`, `x >= 0.`

You will be filling in the core components of the algorithm, including the logic for computing step sizes, solving linear systems for update directions and computing the core measures guiding the algorithm.

## Project Structure

The project is organized into two main packages:

- `src/ipm`: This is your workspace. You will find skeleton files with `TODO` markers indicating where you need to write your implementation.
- `src/ipm`: This directory contains a working implementation of the algorithm. It's there for you to use as the starting implementation. I am happy to take suggestions of improvements if you disapprove of my implementation.

The tests are designed to work with your code in `src/ipm` as you complete the `TODO`s.

## Setup

To get started, you need [uv](https://docs.astral.sh/uv/getting-started/) installed.

1. **Create a virtual environment:**
    From the `linopt/ipm` directory, run:

    ```bash
    uv sync
    ```

    This will create a `.venv` directory with all the dependencies installed.

2. **Open project and select the virtual environment in VSCode**
    Open the folder `linopt/ipm` in VSCode.
    Select interpreter and chose python in the `.venv`.

## Your Task

Your task is to complete the implementation in the `src/ipm` directory. Look for the `TODO` comments in the following files:

- **`ipm_tools.py`**
- **`predictor_corrector.py`**

## Running the Tests

A test suite is provided in the `tests/` directory to help you verify your implementation. The tests cover individual components like the pivoting strategy as well as the full solver on various problems.

To run the tests, make sure your virtual environment is activated and run the following command from the `linopt/ipm` directory:

```bash
pytest .
```

or use the test GUI in VSCode.
To see the log from the solver you can run:

```bash
pytest . -o "log_cli=true" -o "log_cli_level=INFO"
```

The tests are designed to pass one by one as you correctly implement the `TODO`s.

## Resources

The implementation details for the Predictor Corrector algorithm, finding a starting point, and surrounding logic can be found in chapter 14 of *Numerical Optimization*.
