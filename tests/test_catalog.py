"""Tests for catalog ingest, scan_from_catalog, and overrides."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyref.pyref as _pyref_mod
import pytest

if getattr(_pyref_mod, "py_ingest_beamtime", None) is None:
    pytest.skip("catalog not built", allow_module_level=True)

from pyref.io import (
    classify_reflectivity_scan_type,
    get_overrides,
    ingest_beamtime,
    scan_experiment,
    set_override,
)
from pyref.io.catalog_path import resolve_catalog_path
from pyref.io.readers import REQUIRED_SCAN_COLUMNS


def _ensure_flat_beamtime_layout(beamtime: Path) -> None:
    (beamtime / "CCD").mkdir(parents=True, exist_ok=True)


def test_resolve_catalog_path_prefers_new_layout_when_created(tmp_path: Path) -> None:
    beamtime = tmp_path / "beam"
    beamtime.mkdir()
    new_db = tmp_path / ".pyref" / "catalog.db"
    new_db.parent.mkdir(parents=True, exist_ok=True)
    new_db.write_bytes(b"")
    assert resolve_catalog_path(beamtime).resolve() == new_db.resolve()


def test_resolve_catalog_path_ignores_legacy_sidecar(tmp_path: Path) -> None:
    beamtime = tmp_path / "beam"
    beamtime.mkdir()
    legacy = beamtime / ".pyref_catalog.db"
    legacy.write_bytes(b"")
    expected = (tmp_path / ".pyref" / "catalog.db").resolve()
    assert resolve_catalog_path(beamtime).resolve() == expected


def test_resolve_catalog_path_new_wins_when_both_exist(tmp_path: Path) -> None:
    beamtime = tmp_path / "beam"
    beamtime.mkdir()
    new_db = tmp_path / ".pyref" / "catalog.db"
    new_db.parent.mkdir(parents=True, exist_ok=True)
    new_db.write_bytes(b"")
    legacy = beamtime / ".pyref_catalog.db"
    legacy.write_bytes(b"")
    assert resolve_catalog_path(beamtime).resolve() == new_db.resolve()


def test_resolve_catalog_path_defaults_to_new_when_neither_exists(
    tmp_path: Path,
) -> None:
    beamtime = tmp_path / "beam"
    beamtime.mkdir()
    expected = (tmp_path / ".pyref" / "catalog.db").resolve()
    assert resolve_catalog_path(beamtime).resolve() == expected
    assert not expected.exists()


def test_ingest_beamtime_empty_dir(tmp_path: Path) -> None:
    _ensure_flat_beamtime_layout(tmp_path)
    db = ingest_beamtime(tmp_path, incremental=True)
    assert db == resolve_catalog_path(tmp_path)
    assert db.exists()


def test_scan_experiment_catalog_dir_empty(tmp_path: Path) -> None:
    _ensure_flat_beamtime_layout(tmp_path)
    ingest_beamtime(tmp_path, incremental=True)
    lf = scan_experiment(tmp_path)
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    for c in REQUIRED_SCAN_COLUMNS:
        assert c in df.columns, f"missing column {c}"
    assert len(df) == 0


def test_get_overrides_empty(tmp_path: Path) -> None:
    _ensure_flat_beamtime_layout(tmp_path)
    db = ingest_beamtime(tmp_path, incremental=True)
    df = get_overrides(db, path=None)
    assert isinstance(df, pl.DataFrame)
    assert "path" in df.columns
    assert len(df) == 0


def test_set_override_invalid_path_raises(tmp_path: Path) -> None:
    _ensure_flat_beamtime_layout(tmp_path)
    db = ingest_beamtime(tmp_path, incremental=True)
    with pytest.raises(Exception):
        set_override(db, "/nonexistent/path.fits", sample_name="x")


@pytest.fixture(scope="module")
def minimal_fits_dir(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    minimal = fixtures / "minimal.fits"
    if not minimal.is_file():
        return None
    root = tmp_path_factory.mktemp("catalog_fixture")
    ccd = root / "CCD"
    ccd.mkdir()
    (ccd / "minimal.fits").write_bytes(minimal.read_bytes())
    return root


def test_scan_experiment_catalog_has_required_columns(
    minimal_fits_dir: Path | None,
) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    ingest_beamtime(minimal_fits_dir, incremental=False)
    lf = scan_experiment(minimal_fits_dir)
    df = lf.collect()
    for c in REQUIRED_SCAN_COLUMNS:
        assert c in df.columns, f"missing column {c}"


def test_scan_experiment_includes_reflectivity_profile_columns(
    minimal_fits_dir: Path | None,
) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    ingest_beamtime(minimal_fits_dir, incremental=False)
    df = scan_experiment(minimal_fits_dir).collect()
    if len(df) == 0:
        pytest.skip("no rows in catalog")
    assert "reflectivity_profile_index" in df.columns
    assert "reflectivity_scan_type" in df.columns
    assert df["reflectivity_profile_index"].dtype in (pl.Int64, pl.Null)
    assert df["reflectivity_scan_type"].dtype == pl.String


def test_fits_accessor_from_catalog(minimal_fits_dir: Path | None) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    import pyref  # noqa: F401

    ingest_beamtime(minimal_fits_dir, incremental=False)
    df = scan_experiment(minimal_fits_dir).collect()
    if len(df) == 0:
        pytest.skip("no rows in catalog")
    _ = df.fits.img[0]


def test_set_override_bt_scan_point_updates_scan_experiment(
    minimal_fits_dir: Path | None,
) -> None:
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    db = ingest_beamtime(minimal_fits_dir, incremental=False)
    df = scan_experiment(minimal_fits_dir).collect()
    if len(df) == 0:
        pytest.skip("no rows in catalog")
    path = str(df["file_path"][0])
    set_override(db, path, sample_name="corrected_sample")
    df2 = scan_experiment(minimal_fits_dir).collect()
    assert df2["sample_name"][0] == "corrected_sample"


def test_classify_reflectivity_scan_type_fixed_energy() -> None:
    pairs = [(284.0, 0.5 * i) for i in range(10)]
    kind, e_min, e_max, t_min, t_max = classify_reflectivity_scan_type(pairs)
    assert kind == "fixed_energy"
    assert e_min is not None and e_max is not None
    assert t_min is not None and t_max is not None
    assert e_min <= 284.0 <= e_max
    assert t_min <= 0.5 and t_max >= 4.5


def test_classify_reflectivity_scan_type_fixed_angle() -> None:
    pairs = [(250.0 + i, 10.0) for i in range(10)]
    kind, _, _, _, _ = classify_reflectivity_scan_type(pairs)
    assert kind == "fixed_angle"
