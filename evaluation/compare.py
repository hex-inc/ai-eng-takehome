"""Compare SQL result multisets while allowing extra columns and numeric tolerance."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from decimal import Decimal, localcontext
from itertools import product

import polars as pl

DEFAULT_EPSILON = 1e-4
type Cell = tuple[str, Decimal | str]
type Row = tuple[Cell, ...]


def _cell(value: object) -> Cell:
    """Keep types distinct while making integer, decimal, and float values comparable."""
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", str(value))
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
        if number.is_nan():
            return ("nan", "")
        if number.is_infinite():
            return ("infinity", str(number))
        return ("number", number)
    return (type(value).__name__, str(value))


def _equal(left: Cell, right: Cell, tolerance: Decimal) -> bool:
    """Compare finite numbers with absolute tolerance and all other values exactly."""
    if left[0] != right[0]:
        return False
    if isinstance(left[1], Decimal) and isinstance(right[1], Decimal):
        # DuckDB decimals may have 38 digits, beyond Python's default precision of 28.
        with localcontext() as context:
            context.prec = 80
            return abs(left[1] - right[1]) <= tolerance
    return left == right


def _row_equal(left: Row, right: Row, tolerance: Decimal) -> bool:
    """Check corresponding cells of two rows without losing their associations."""
    return all(_equal(a, b, tolerance) for a, b in zip(left, right, strict=True))


def _match_rows(expected: list[Row], actual: list[Row], tolerance: Decimal) -> bool:
    """Find a one-to-one row match, including ambiguous overlapping tolerance windows."""
    if all(
        _row_equal(a, b, tolerance) for a, b in zip(sorted(expected), sorted(actual), strict=True)
    ):
        return True

    # Exact nonnumeric cells restrict candidate rows before numeric comparisons.
    buckets: dict[tuple[Cell, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(actual):
        key = tuple(cell if cell[0] != "number" else ("number", "") for cell in row)
        buckets[key].append(index)
    edges = []
    for row in expected:
        key = tuple(cell if cell[0] != "number" else ("number", "") for cell in row)
        matches = [index for index in buckets[key] if _row_equal(row, actual[index], tolerance)]
        if not matches:
            return False
        edges.append(matches)

    # Augmenting paths can reassign earlier matches; greedy matching is not sufficient.
    owners: dict[int, int] = {}
    for start in range(len(expected)):
        queue = deque([start])
        parents: dict[int, tuple[int, int] | None] = {start: None}
        endpoint: tuple[int, int] | None = None
        while queue and endpoint is None:
            left = queue.popleft()
            for right in edges[left]:
                if right not in owners:
                    endpoint = (left, right)
                    break
                owner = owners[right]
                if owner not in parents:
                    parents[owner] = (left, right)
                    queue.append(owner)
        if endpoint is None:
            return False
        while endpoint is not None:
            left, right = endpoint
            owners[right] = left
            endpoint = parents[left]
    return True


def loosely_compare_dataframes(
    gold_df: pl.DataFrame,
    submitted_df: pl.DataFrame,
    epsilon: float = DEFAULT_EPSILON,
) -> bool:
    """Compare row multisets, ignoring column names, order, and extra submitted columns.

    Numeric types are interchangeable within an absolute tolerance (default 0.0001).
    Duplicate rows and associations between columns must match. NULL, NaN, infinity,
    booleans, and strings remain distinct. Two empty results match if the submitted
    result has at least the required number of columns.
    """
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if gold_df.height != submitted_df.height or submitted_df.width < gold_df.width:
        return False
    if gold_df.height == 0:
        return True

    tolerance = Decimal(str(epsilon))
    gold_columns = [[_cell(value) for value in column] for column in gold_df.iter_columns()]
    actual_columns = [[_cell(value) for value in column] for column in submitted_df.iter_columns()]
    sorted_actual = [sorted(column) for column in actual_columns]
    candidates = []
    for column in gold_columns:
        expected = sorted(column)
        matches = [
            index
            for index, actual in enumerate(sorted_actual)
            if all(_equal(a, b, tolerance) for a, b in zip(expected, actual, strict=True))
        ]
        if not matches:
            return False
        candidates.append(matches)

    gold_rows = list(zip(*gold_columns, strict=True))
    for assignment in product(*candidates):
        if len(set(assignment)) != len(assignment):
            continue
        actual_rows = list(zip(*(actual_columns[index] for index in assignment), strict=True))
        if _match_rows(gold_rows, actual_rows, tolerance):
            return True
    return False
