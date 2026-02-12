"""Tests for df.fits DataFrame accessor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyref  # noqa: F401 - registers df.fits accessor
import pytest

HEADER_KEYS = ["Sample Theta"]
REQUIRED_COLUMNS = (
    "file_path",
    "data_offset",
    "naxis1",
    "naxis2",
    "bitpix",
    "bzero",
    "file_name",
)


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "minimal.fits"


@pytest.fixture(scope="module")
def minimal_fits_path() -> Path:
    """Return path to minimal FITS fixture."""
    p = _fixture_path()
    if not p.is_file():
        pytest.skip(f"Minimal FITS fixture not found: {p}")
    return p


@pytest.fixture(scope="module")
def fits_meta_df(minimal_fits_path: Path) -> pl.DataFrame:
    """Return FITS metadata DataFrame from minimal fixture."""
    try:
        from pyref.pyref import py_read_fits_headers_only
    except ImportError:
        pytest.skip("pyref extension not available")
    df = py_read_fits_headers_only(str(minimal_fits_path), HEADER_KEYS)
    assert all(c in df.columns for c in REQUIRED_COLUMNS)
    return df


def test_fits_accessor_schema_validation() -> None:
    """Schema validation rejects DataFrame without FITS metadata columns."""
    bad_df = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Missing columns"):
        _ = bad_df.fits.img[0]


def test_fits_img_single(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.img[i] returns raw image ndarray."""
    img = fits_meta_df.fits.img[0]
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.dtype.kind in ("i", "u")


def test_fits_img_len(fits_meta_df: pl.DataFrame) -> None:
    """len(df.fits.img) equals len(df)."""
    assert len(fits_meta_df.fits.img) == len(fits_meta_df)


def test_fits_img_slice(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.img[slice] returns iterator of raw images."""
    it = fits_meta_df.fits.img[:1]
    imgs = list(it)
    assert len(imgs) == 1
    assert imgs[0].ndim == 2


def test_fits_img_iter(fits_meta_df: pl.DataFrame) -> None:
    """Iterating df.fits.img yields raw images."""
    imgs = list(fits_meta_df.fits.img)
    assert len(imgs) == len(fits_meta_df)
    assert all(isinstance(x, np.ndarray) and x.ndim == 2 for x in imgs)


def test_fits_corrected_single(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.corrected(i) returns bg-corrected image."""
    import importlib.util

    if importlib.util.find_spec("pyref.pyref") is None:
        pytest.skip("pyref extension not available")
    try:
        from pyref.pyref import py_get_image_corrected

        _ = py_get_image_corrected
    except ImportError:
        pytest.skip("py_get_image_corrected not available")
    arr = fits_meta_df.fits.corrected(0, bg_rows=5, bg_cols=5)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2


def test_fits_corrected_slice(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.corrected(slice) returns iterator of bg-corrected images."""
    import importlib.util

    if importlib.util.find_spec("pyref.pyref") is None:
        pytest.skip("pyref extension not available")
    try:
        from pyref.pyref import py_get_image_corrected

        _ = py_get_image_corrected
    except ImportError:
        pytest.skip("py_get_image_corrected not available")
    it = fits_meta_df.fits.corrected(slice(0, 1), bg_rows=5, bg_cols=5)
    arrs = list(it)
    assert len(arrs) == 1
    assert arrs[0].ndim == 2


def test_fits_filtered_single(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.filtered(i, sigma) returns bg-corrected and gaussian-filtered image."""
    import importlib.util

    if importlib.util.find_spec("pyref.pyref") is None:
        pytest.skip("pyref extension not available")
    try:
        from pyref.pyref import py_materialize_image_filtered_edges

        _ = py_materialize_image_filtered_edges
    except ImportError:
        pytest.skip("py_materialize_image_filtered_edges not available")
    arr = fits_meta_df.fits.filtered(0, 1.5, bg_rows=5, bg_cols=5)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")


def test_fits_filtered_slice(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.filtered(slice, sigma) returns iterator of filtered images."""
    import importlib.util

    if importlib.util.find_spec("pyref.pyref") is None:
        pytest.skip("pyref extension not available")
    try:
        from pyref.pyref import py_materialize_image_filtered_edges

        _ = py_materialize_image_filtered_edges
    except ImportError:
        pytest.skip("py_materialize_image_filtered_edges not available")
    it = fits_meta_df.fits.filtered(slice(0, 1), 1.5, bg_rows=5, bg_cols=5)
    arrs = list(it)
    assert len(arrs) == 1
    assert arrs[0].dtype == np.dtype("float32")


def test_fits_custom_single(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.custom(i, callable) applies callable to image."""
    def identity(img: np.ndarray) -> np.ndarray:
        return img

    result = fits_meta_df.fits.custom(0, identity)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_fits_custom_slice(fits_meta_df: pl.DataFrame) -> None:
    """df.fits.custom(slice, callable) applies callable to images, returns iterator."""
    def double(img: np.ndarray) -> np.ndarray:
        return img * 2

    it = fits_meta_df.fits.custom(slice(0, 1), double)
    results = list(it)
    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)
