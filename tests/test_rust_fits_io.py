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


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "minimal.fits"


@pytest.fixture(scope="module")
def minimal_fits_path() -> Path:
    p = _fixture_path()
    if not p.is_file():
        pytest.skip(f"Minimal FITS fixture not found: {p}")
    return p


def test_rust_read_fits_minimal_fixture(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits

    df = py_read_fits(str(minimal_fits_path), HEADER_KEYS)
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    assert "file_name" in df.columns
    assert "RAW" in df.columns
    assert df.schema["file_name"] == pl.String
    stem = df["file_name"][0]
    assert stem == "minimal"


def test_rust_read_fits_minimal_empty_headers(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits

    df = py_read_fits(str(minimal_fits_path), [])
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    assert "file_name" in df.columns
    assert "RAW" in df.columns


def test_rust_read_fits_parsed_columns(minimal_fits_path: Path) -> None:
    from pyref.pyref import py_read_fits

    df = py_read_fits(str(minimal_fits_path), HEADER_KEYS)
    for col in ("file_name", "sample_name", "tag", "experiment_number", "frame_number"):
        assert col in df.columns
