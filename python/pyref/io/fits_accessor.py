"""
FITS DataFrame accessor for image loading from scan_experiment metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
import polars as pl

from pyref.io.readers import (
    REQUIRED_SCAN_COLUMNS,
    get_image,
    get_image_corrected,
    get_image_filtered_edges,
)


def _validate_fits_schema(df: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_SCAN_COLUMNS if c not in df.columns]
    if missing:
        msg = (
            "DataFrame must be from scan_experiment().collect() with FITS metadata "
            f"schema. Missing columns: {missing}"
        )
        raise ValueError(msg)


def _indices_from_slice(s: slice, n: int) -> range:
    start, stop, step = s.indices(n)
    return range(start, stop, step)


@pl.api.register_dataframe_namespace("fits")
class FitsFrame:
    """DataFrame accessor for FITS image loading from scan_experiment metadata."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    @property
    def img(self) -> _FitsImageIndexer:
        """Indexer for raw detector images."""
        return _FitsImageIndexer(self._df)

    def corrected(
        self,
        idx: int | slice,
        bg_rows: int = 10,
        bg_cols: int = 10,
    ) -> np.ndarray | _FitsCorrectedIterator:
        """Return background-corrected image(s) at idx."""
        _validate_fits_schema(self._df)
        if isinstance(idx, int):
            arr = get_image_corrected(self._df, idx, bg_rows, bg_cols)
            return np.asarray(arr)
        indices = _indices_from_slice(idx, len(self._df))
        return _FitsCorrectedIterator(self._df, indices, bg_rows, bg_cols)

    def filtered(
        self,
        idx: int | slice,
        sigma: float,
        bg_rows: int = 10,
        bg_cols: int = 10,
    ) -> np.ndarray | _FitsFilteredIterator:
        """Return background-corrected and gaussian-filtered image(s) at idx."""
        _validate_fits_schema(self._df)
        if isinstance(idx, int):
            arr = get_image_filtered_edges(
                self._df, idx, sigma, bg_rows, bg_cols
            )
            return np.asarray(arr)
        indices = _indices_from_slice(idx, len(self._df))
        return _FitsFilteredIterator(
            self._df, indices, sigma, bg_rows, bg_cols
        )

    def custom(
        self,
        idx: int | slice,
        callable_fn: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        """Apply callable to image(s) at idx."""
        _validate_fits_schema(self._df)
        if isinstance(idx, int):
            img = get_image(self._df, idx)
            return callable_fn(np.asarray(img), **kwargs)
        indices = _indices_from_slice(idx, len(self._df))
        return _FitsCustomIterator(self._df, indices, callable_fn, kwargs)


class _FitsImageIndexer:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key: int | slice) -> np.ndarray | _FitsImgIterator:
        _validate_fits_schema(self._df)
        if isinstance(key, int):
            arr = get_image(self._df, key)
            return np.asarray(arr)
        indices = _indices_from_slice(key, len(self._df))
        return _FitsImgIterator(self._df, indices)

    def __iter__(self) -> _FitsImgIterator:
        return _FitsImgIterator(self._df, range(len(self._df)))


class _FitsImgIterator:
    def __init__(self, df: pl.DataFrame, indices: range) -> None:
        self._df = df
        self._iter = iter(indices)

    def __iter__(self) -> _FitsImgIterator:
        return self

    def __next__(self) -> np.ndarray:
        i = next(self._iter)
        arr = get_image(self._df, i)
        return np.asarray(arr)


class _FitsCorrectedIterator:
    def __init__(
        self,
        df: pl.DataFrame,
        indices: range,
        bg_rows: int,
        bg_cols: int,
    ) -> None:
        self._df = df
        self._indices_iter = iter(indices)
        self._bg_rows = bg_rows
        self._bg_cols = bg_cols

    def __iter__(self) -> _FitsCorrectedIterator:
        return self

    def __next__(self) -> np.ndarray:
        i = next(self._indices_iter)
        arr = get_image_corrected(
            self._df, i, self._bg_rows, self._bg_cols
        )
        return np.asarray(arr)


class _FitsFilteredIterator:
    def __init__(
        self,
        df: pl.DataFrame,
        indices: range,
        sigma: float,
        bg_rows: int,
        bg_cols: int,
    ) -> None:
        self._df = df
        self._indices_iter = iter(indices)
        self._sigma = sigma
        self._bg_rows = bg_rows
        self._bg_cols = bg_cols

    def __iter__(self) -> _FitsFilteredIterator:
        return self

    def __next__(self) -> np.ndarray:
        i = next(self._indices_iter)
        arr = get_image_filtered_edges(
            self._df, i, self._sigma, self._bg_rows, self._bg_cols
        )
        return np.asarray(arr)


class _FitsCustomIterator:
    def __init__(
        self,
        df: pl.DataFrame,
        indices: range,
        callable_fn: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> None:
        self._df = df
        self._indices_iter = iter(indices)
        self._callable = callable_fn
        self._kwargs = kwargs

    def __iter__(self) -> _FitsCustomIterator:
        return self

    def __next__(self) -> object:
        i = next(self._indices_iter)
        img = get_image(self._df, i)
        return self._callable(np.asarray(img), **self._kwargs)
