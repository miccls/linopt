import gzip
import json
import os
import tempfile
import urllib.request
import warnings
from collections.abc import Iterable
from typing import cast

import numpy as np
import pulp
from common.numpy_type_aliases import ArrayF


CONVERTED_NETLIB_BASE_URL = (
    "https://raw.githubusercontent.com/ozy4dm/lp-data-netlib/main/mps_files"
)
COIN_OR_NETLIB_BASE_URL = (
    "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master"
)
CONVERTED_NETLIB_API_URL = (
    "https://api.github.com/repos/ozy4dm/lp-data-netlib/contents/mps_files"
)


def list_available_netlib_problems() -> list[str]:
    """Return problem names available in the converted Netlib MPS mirror."""
    with urllib.request.urlopen(CONVERTED_NETLIB_API_URL) as response:
        entries = json.load(response)

    return sorted(
        entry["name"].removesuffix(".mps")
        for entry in entries
        if entry["name"].endswith(".mps")
    )


def _candidate_urls(problem_name: str) -> Iterable[tuple[str, bool]]:
    yield f"{CONVERTED_NETLIB_BASE_URL}/{problem_name}.mps", False
    yield f"{COIN_OR_NETLIB_BASE_URL}/{problem_name}.mps.gz", True


def _download_mps(problem_name: str) -> str:
    errors: list[str] = []
    for url, is_gzip in _candidate_urls(problem_name):
        print(f"Downloading {problem_name} from {url}...")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response:
                data = cast(bytes, response.read())
            if is_gzip:
                return gzip.decompress(data).decode("utf-8")
            return data.decode("utf-8")
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")

    raise RuntimeError(
        f"Failed to download Netlib problem {problem_name}. Tried:\n"
        + "\n".join(errors)
    )


def _normalize_mps_for_pulp(mps_text: str) -> str:
    """Add synthetic RHS/BOUNDS names for valid no-name sections PuLP cannot read."""
    normalized_lines: list[str] = []
    section: str | None = None
    variable_names: set[str] = set()

    for line in mps_text.splitlines():
        fields = line.split()
        if not fields:
            normalized_lines.append(line)
            continue

        keyword = fields[0]
        if keyword in {"ROWS", "COLUMNS", "RHS", "BOUNDS"}:
            section = keyword
            normalized_lines.append(line)
            continue
        if keyword == "ENDATA":
            section = None
            normalized_lines.append(line)
            continue
        if keyword.startswith("*"):
            normalized_lines.append(line)
            continue

        if section == "COLUMNS":
            variable_names.add(fields[0])

        if section == "RHS" and len(fields) in {2, 4}:
            normalized_lines.append("    RHS1      " + " ".join(fields))
        elif (
            section == "BOUNDS"
            and len(fields) in {2, 3}
            and fields[1] in variable_names
        ):
            normalized_lines.append(
                f" {fields[0]:<2} BND1      " + " ".join(fields[1:])
            )
        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines) + "\n"


def _parse_mps_with_pulp(mps_text: str) -> pulp.LpProblem:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".mps", delete=False)
    try:
        with tmp:
            tmp.write(_normalize_mps_for_pulp(mps_text))
        _, model = pulp.LpProblem.fromMPS(tmp.name, sense=pulp.LpMinimize)
        return model
    finally:
        os.unlink(tmp.name)


def _row_type_from_pulp_sense(sense: int) -> str:
    if sense == pulp.LpConstraintEQ:
        return "E"
    if sense == pulp.LpConstraintLE:
        return "L"
    if sense == pulp.LpConstraintGE:
        return "G"
    raise ValueError(f"Unsupported PuLP constraint sense: {sense}")


def _append_bound_rows(
    a_rows: list[dict[int, float]],
    b_values: list[float],
    row_types: list[str],
    variables: list[pulp.LpVariable],
) -> None:
    for col_idx, var in enumerate(variables):
        lower = var.lowBound
        upper = var.upBound

        if lower is None or lower < 0.0:
            warnings.warn(
                f"Column {col_idx} ({var.name}) is not nonnegative. "
                "Standard form conversion requires variable splitting.",
                stacklevel=2,
            )

        if lower is not None and upper is not None and lower == upper:
            a_rows.append({col_idx: 1.0})
            b_values.append(float(lower))
            row_types.append("E")
            continue

        if upper is not None:
            a_rows.append({col_idx: 1.0})
            b_values.append(float(upper))
            row_types.append("L")

        if lower is not None and lower != 0.0:
            a_rows.append({col_idx: 1.0})
            b_values.append(float(lower))
            row_types.append("G")


def _model_to_matrices(
    model: pulp.LpProblem,
) -> tuple[ArrayF, ArrayF, ArrayF, list[str], dict[str, int], dict[str, int]]:
    variables = model.variables()
    col_map = {var.name: col_idx for col_idx, var in enumerate(variables)}
    row_map: dict[str, int] = {}
    row_types: list[str] = []
    a_rows: list[dict[int, float]] = []
    b_values: list[float] = []

    for row_idx, constraint in enumerate(model.constraints()):
        name = constraint.name or f"row_{row_idx}"
        row_map[name] = row_idx
        row_types.append(_row_type_from_pulp_sense(constraint.sense))
        b_values.append(float(-constraint.constant))
        a_rows.append(
            {col_map[var.name]: float(value) for var, value in constraint.items()}
        )

    base_num_rows = len(a_rows)
    _append_bound_rows(a_rows, b_values, row_types, variables)

    a: ArrayF = np.zeros((len(a_rows), len(variables)))
    b: ArrayF = np.array(b_values, dtype=float)
    c: ArrayF = np.zeros(len(variables))

    for row_idx, row in enumerate(a_rows):
        for col_idx, value in row.items():
            a[row_idx, col_idx] = value

    for var, value in model.objective.items():
        c[col_map[var.name]] = float(value)

    print(
        f"Parsed successfully: Base a is {base_num_rows}x{len(variables)}, "
        f"plus {len(a_rows) - base_num_rows} bound rows."
    )
    return a, b, c, row_types, col_map, row_map


def download_and_parse_mps(
    problem_name: str,
) -> (
    tuple[
        ArrayF,
        ArrayF,
        ArrayF,
        list[str],
        dict[str, int],
        dict[str, int],
    ]
    | None
):
    try:
        mps_text = _download_mps(problem_name)
        model = _parse_mps_with_pulp(mps_text)
        return _model_to_matrices(model)
    except Exception as e:
        print(f"Failed to download/parse {problem_name}: {e}")
        return None


def convert_to_standard_form(
    a: ArrayF, b: ArrayF, c: ArrayF, row_types: list[str]
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """
    Converts mixed inequalities into standard form (Ax = b)
    by appending slack and surplus variables.
    """
    m, n = a.shape

    num_slacks = sum(1 for rtype in row_types if rtype in ["L", "G"])

    a_std: ArrayF = np.zeros((m, n + num_slacks))
    c_std: ArrayF = np.zeros(n + num_slacks)

    a_std[:, :n] = a
    c_std[:n] = c

    slack_idx = n
    for i, rtype in enumerate(row_types):
        if rtype == "L":
            a_std[i, slack_idx] = 1.0
            slack_idx += 1
        elif rtype == "G":
            a_std[i, slack_idx] = -1.0
            slack_idx += 1

    print(f"Added {num_slacks} slack/surplus variables.")
    print(f"New a shape: {a_std.shape}, New c shape: {c_std.shape}")

    return a_std, b, c_std


if __name__ == "__main__":
    result = download_and_parse_mps("afiro")

    if result:
        a_res, b_res, c_res, row_types_res, col_map_res, row_map_res = result
        print("\nConstraint Matrix (a) shape:", a_res.shape)
        print("Right-Hand Side (b) shape:", b_res.shape)
        print("Objective Vector (c) shape:", c_res.shape)
        print("\nFirst 5 objective coefficients:", c_res[:5])
