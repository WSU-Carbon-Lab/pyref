"""Tests for Rust FITS I/O using minimal fixture (no external data)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

HEADER_KEYS = [
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
]

REQUIRED_HEADER_ONLY_COLUMNS = (
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
    p = _fixture_path()
    if not p.is_file():
        pytest.skip(f"Minimal FITS fixture not found: {p}")
    return p


def test_rust_read_fits_headers_only_minimal_fixture(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), HEADER_KEYS)
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    for col in REQUIRED_HEADER_ONLY_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    assert df.schema["file_name"] == pl.String
    stem = df["file_name"][0]
    assert stem == "minimal"


def test_rust_read_fits_headers_only_empty_headers(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), [])
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    for col in REQUIRED_HEADER_ONLY_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_rust_read_fits_headers_only_parsed_columns(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), HEADER_KEYS)
    for col in ("file_name", "sample_name", "tag", "experiment_number", "frame_number"):
        assert col in df.columns


def test_rust_read_fits_headers_only_image_ref_columns(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), HEADER_KEYS)
    assert "file_path" in df.columns
    assert "data_offset" in df.columns
    assert "naxis1" in df.columns
    assert "naxis2" in df.columns
    assert "bitpix" in df.columns
    assert "bzero" in df.columns
    row = df.row(0, named=True)
    assert row["naxis1"] > 0 and row["naxis2"] > 0
    assert isinstance(row["file_path"], str)
    assert isinstance(row["data_offset"], int)


def test_rust_read_multiple_fits_headers_only_single(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_multiple_fits_headers_only

    df = py_read_multiple_fits_headers_only([str(minimal_fits_path)], HEADER_KEYS)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    for col in REQUIRED_HEADER_ONLY_COLUMNS:
        assert col in df.columns


def test_rust_get_image_for_row(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_get_image_for_row, py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), [])
    raw, subtracted = py_get_image_for_row(df, 0)
    import numpy as np

    assert hasattr(raw, "shape")
    assert hasattr(subtracted, "shape")
    raw_arr = np.asarray(raw)
    sub_arr = np.asarray(subtracted)
    assert raw_arr.ndim == 2 and sub_arr.ndim == 2
    assert raw_arr.shape == sub_arr.shape
    assert raw_arr.dtype.kind in ("i", "u") and sub_arr.dtype.kind in ("i", "u")


def test_rust_materialize_image_filtered(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_materialize_image_filtered, py_read_fits_headers_only

    df = py_read_fits_headers_only(str(minimal_fits_path), [])
    blurred = py_materialize_image_filtered(df, 0, 1.5)
    import numpy as np

    arr = np.asarray(blurred)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")
