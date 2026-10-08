from decimal import Decimal

import polars as pl
import pytest

from evaluation.compare import loosely_compare_dataframes


def test_decimal_results_match_equivalent_numeric_types() -> None:
    """Accept equivalent decimal, float, and integer values but reject changed values."""
    expected = pl.DataFrame(
        {
            "abv": [Decimal("0.095"), Decimal("0.100"), None],
            "count": [Decimal("1.000"), Decimal("2.000"), Decimal("0.000")],
        }
    )
    submitted = pl.DataFrame({"abv": [0.095, 0.1, None], "count": [1, 2, 0]})

    assert loosely_compare_dataframes(expected, submitted)
    assert not loosely_compare_dataframes(
        expected, submitted.with_columns((pl.col("abv") - 0.001).alias("abv"))
    )


def test_absolute_tolerance_does_not_depend_on_rounding_boundaries() -> None:
    """Accept nearby values across a rounding boundary, but reject larger differences."""
    expected = pl.DataFrame({"value": [0.12349]})
    assert loosely_compare_dataframes(expected, pl.DataFrame({"value": [0.12351]}))
    assert not loosely_compare_dataframes(expected, pl.DataFrame({"value": [0.1231]}))
    assert not loosely_compare_dataframes(expected, pl.DataFrame({"value": [0.12351]}), 0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        loosely_compare_dataframes(expected, expected, float("nan"))


def test_column_permutations_preserve_rows_and_duplicate_counts() -> None:
    """Extra columns are allowed, but matching individual columns is not enough."""
    expected = pl.DataFrame({"name": ["a", "b", "b"], "value": [1, 2, 2]})
    actual = pl.DataFrame({"amount": [2.0, 1.0, 2.0], "extra": [0, 0, 0], "label": ["b", "a", "b"]})
    assert loosely_compare_dataframes(expected, actual)
    assert not loosely_compare_dataframes(
        expected, actual.with_columns(pl.Series("label", ["a", "b", "b"]))
    )
    assert not loosely_compare_dataframes(expected, actual.head(2))


def test_overlapping_tolerance_windows_allow_reassignment() -> None:
    """Multicolumn tolerance matching must recover when sorted or greedy pairing fails."""
    expected = pl.DataFrame({"x": [0.0, 0.0001], "y": [1.0, 2.0]})
    actual = pl.DataFrame({"a": [0.0001, 0.0], "b": [1.0, 2.0]})
    assert loosely_compare_dataframes(expected, actual)
    # The first expected row can match both; the second can only match the first actual row.
    expected = pl.DataFrame({"x": [0.0, 0.0002], "y": [0.0, 0.0002]})
    actual = pl.DataFrame({"x": [0.0001, -0.0001], "y": [0.0001, 0.0001]})
    assert loosely_compare_dataframes(expected, actual)


def test_special_values_and_large_decimals_are_not_string_sentinels() -> None:
    """Keep SQL NULLs, strings, nonfinite numbers, and exact large decimals distinct."""
    expected = pl.DataFrame({"n": [None, float("nan"), float("inf"), float("-inf")]})
    assert loosely_compare_dataframes(expected, expected.reverse())
    assert not loosely_compare_dataframes(
        expected, pl.DataFrame({"n": ["__NULL__", "__NAN__", "__INF__", "__NEG_INF__"]})
    )
    large = Decimal("123456789012345678901234567890123456.78")
    assert loosely_compare_dataframes(pl.DataFrame({"n": [large]}), pl.DataFrame({"n": [large]}))
    assert not loosely_compare_dataframes(pl.DataFrame({"n": [1]}), pl.DataFrame({"n": ["1"]}))


def test_empty_results_still_require_the_requested_columns() -> None:
    """Accept valid zero-row results without allowing missing output columns."""
    expected = pl.DataFrame(schema={"id": pl.Int64, "value": pl.Float64})
    assert loosely_compare_dataframes(expected, expected)
    assert not loosely_compare_dataframes(expected, pl.DataFrame(schema={"id": pl.Int64}))
