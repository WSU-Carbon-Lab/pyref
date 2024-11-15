"""Python utilities for Polars."""

from pathlib import Path

import polars as pl
from polars.datatypes import DataType, DataTypeClass
from polars.plugins import register_plugin_function

# from pyref.utils.file_dialogs import FileDialog

type IntoExprColumn = pl.Expr | str | pl.Series
type PolarsDataType = DataType | DataTypeClass

LIB = Path(__file__).parent.parent

# ==================/ Statistics imports /==================


def weighted_mean(expr: IntoExprColumn, weights: IntoExprColumn) -> pl.Expr:
    """Calculate the weighted mean of a column."""
    return register_plugin_function(
        args=[expr, weights],
        plugin_path=LIB,
        function_name="weighted_mean",
        is_elementwise=True,
    )


def weighted_std(expr: IntoExprColumn, weights: IntoExprColumn) -> pl.Expr:
    """Calculate the weighted standard deviation of a column."""
    return register_plugin_function(
        args=[expr, weights],
        plugin_path=LIB,
        function_name="weighted_std",
        is_elementwise=True,
    )


def err_prop_mult(
    lhs: IntoExprColumn,
    lhs_err: IntoExprColumn,
    rhs: IntoExprColumn,
    rhs_err: IntoExprColumn,
) -> pl.Expr:
    """Calculate the error propagation for multiplication."""
    return register_plugin_function(
        args=[lhs, lhs_err, rhs, rhs_err],
        plugin_path=LIB,
        function_name="err_prop_mult",
        is_elementwise=True,
    )


def err_prop_div(
    lhs: IntoExprColumn,
    lhs_err: IntoExprColumn,
    rhs: IntoExprColumn,
    rhs_err: IntoExprColumn,
) -> pl.Expr:
    """Calculate the error propagation for division."""
    return register_plugin_function(
        args=[lhs, lhs_err, rhs, rhs_err],
        plugin_path=LIB,
        function_name="err_prop_div",
        is_elementwise=True,
    )


__all__ = ["weighted_mean", "weighted_std"]  # "FileDialog"]
