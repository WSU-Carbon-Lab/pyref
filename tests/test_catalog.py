"""Tests for catalog ingest, scan_from_catalog, and overrides."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import pyref.pyref as _pyref_mod
if getattr(_pyref_mod, "py_ingest_beamtime", None) is None:
    pytest.skip("catalog not built", allow_module_level=True)

from pyref.io.readers import REQUIRED_SCAN_COLUMNS
from pyref.io import get_overrides, ingest_beamtime, scan_experiment, set_override


def test_ingest_beamtime_empty_dir(tmp_path: Path) -> None:
    db = ingest_beamtime(tmp_path, incremental=True)
    assert db == tmp_path / ".pyref_catalog.db"
    assert db.exists()


def test_scan_experiment_catalog_dir_empty(tmp_path: Path) -> None:
    ingest_beamtime(tmp_path, incremental=True)
    lf = scan_experiment(tmp_path)
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    for c in REQUIRED_SCAN_COLUMNS:
        assert c in df.columns, f"missing column {c}"
    assert len(df) == 0


def test_get_overrides_empty(tmp_path: Path) -> None:
    ingest_beamtime(tmp_path, incremental=True)
    db = tmp_path / ".pyref_catalog.db"
    df = get_overrides(db, path=None)
    assert isinstance(df, pl.DataFrame)
    assert "path" in df.columns
    assert len(df) == 0


def test_set_override_invalid_path_raises(tmp_path: Path) -> None:
    ingest_beamtime(tmp_path, incremental=True)
    db = tmp_path / ".pyref_catalog.db"
    with pytest.raises(Exception):
        set_override(db, "/nonexistent/path.fits", sample_name="x")


@pytest.fixture(scope="module")
def minimal_fits_dir(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    minimal = fixtures / "minimal.fits"
    if not minimal.is_file():
        return None
    root = tmp_path_factory.mktemp("catalog_fixture")
    (root / "minimal.fits").write_bytes(minimal.read_bytes())
    return root


def test_scan_experiment_catalog_has_required_columns(minimal_fits_dir: Path | None) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    ingest_beamtime(minimal_fits_dir, incremental=False)
    lf = scan_experiment(minimal_fits_dir)
    df = lf.collect()
    for c in REQUIRED_SCAN_COLUMNS:
        assert c in df.columns, f"missing column {c}"


def test_fits_accessor_from_catalog(minimal_fits_dir: Path | None) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    import pyref  # noqa: F401
    ingest_beamtime(minimal_fits_dir, incremental=False)
    df = scan_experiment(minimal_fits_dir).collect()
    if len(df) == 0:
        pytest.skip("no rows in catalog")
    _ = df.fits.img[0]
