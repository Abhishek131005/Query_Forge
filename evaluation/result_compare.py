"""Deterministic comparison of SQLite result sets for benchmark scoring."""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd


FLOAT_TOLERANCE = 1e-9


def _value_key(value):
    if pd.isna(value):
        return ("null",)
    if isinstance(value, float):
        return ("number", round(value, 9))
    return (type(value).__name__, str(value))


def _row_key(row):
    return tuple(_value_key(value) for value in row)


def compare_results(expected: pd.DataFrame, actual: pd.DataFrame, order_sensitive: bool) -> bool:
    """Compare columns, duplicates, NULLs, numeric values, and row ordering."""
    if list(expected.columns) != list(actual.columns):
        return False
    if expected.shape[0] != actual.shape[0]:
        return False
    if expected.empty:
        return True

    expected_rows = [_row_key(row) for row in expected.itertuples(index=False, name=None)]
    actual_rows = [_row_key(row) for row in actual.itertuples(index=False, name=None)]
    if order_sensitive:
        return expected_rows == actual_rows
    return Counter(expected_rows) == Counter(actual_rows)
