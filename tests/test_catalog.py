"""Tests for catalog ingest, scan_from_catalog, and overrides."""

# ruff: noqa: D103, PT018

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyref.pyref as _pyref_mod
import pytest

if getattr(_pyref_mod, "py_ingest_beamtime", None) is None:
    pytest.skip("catalog not built", allow_module_level=True)

if getattr(_pyref_mod, "py_default_catalog_db_path", None) is None:
    pytest.skip("catalog path export missing", allow_module_level=True)

from pyref.io import (
    beamtime_ingest_layout,
    classify_reflectivity_scan_type,
    get_overrides,
    ingest_beamtime,
    list_beamtimes,
    read_beamtime,
    scan_experiment,
    scan_from_catalog_for_beamtime,
    set_override,
)
from pyref.io.catalog_path import resolve_catalog_path
from pyref.io.readers import REQUIRED_SCAN_COLUMNS


@pytest.fixture(autouse=True)
def _pyref_home_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYREF_HOME", str(tmp_path))


def _ensure_flat_beamtime_layout(beamtime: Path) -> None:
    (beamtime / "CCD").mkdir(parents=True, exist_ok=True)


def test_resolve_catalog_path_global(tmp_path: Path) -> None:
    expected = (tmp_path / "catalog.db").resolve()
    assert resolve_catalog_path().resolve() == expected
    assert resolve_catalog_path(tmp_path / "any" / "beam").resolve() == expected


def test_ingest_beamtime_empty_dir(tmp_path: Path) -> None:
    _ensure_flat_beamtime_layout(tmp_path)
    db = ingest_beamtime(tmp_path, incremental=True)
    assert db == resolve_catalog_path()
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
    with pytest.raises(RuntimeError, match="path not in catalog"):
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


def test_set_override_updates_scan_experiment(
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


def test_beamtime_ingest_layout_empty_ccd(tmp_path: Path) -> None:
    if getattr(_pyref_mod, "py_beamtime_ingest_layout", None) is None:
        pytest.skip("py_beamtime_ingest_layout not built")
    _ensure_flat_beamtime_layout(tmp_path)
    layout = beamtime_ingest_layout(tmp_path)
    assert layout["total_files"] == 0
    assert layout["scans"] == []


def test_beamtime_ingest_layout_minimal(minimal_fits_dir: Path | None) -> None:
    if getattr(_pyref_mod, "py_beamtime_ingest_layout", None) is None:
        pytest.skip("py_beamtime_ingest_layout not built")
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    layout = beamtime_ingest_layout(minimal_fits_dir)
    assert layout["total_files"] >= 1
    assert len(layout["scans"]) >= 1
    assert all("scan_number" in s and "files" in s for s in layout["scans"])


def test_ingest_beamtime_progress_callback(minimal_fits_dir: Path | None) -> None:
    if getattr(_pyref_mod, "py_beamtime_ingest_layout", None) is None:
        pytest.skip("py_beamtime_ingest_layout not built")
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    events: list[dict[str, object]] = []
    ingest_beamtime(
        minimal_fits_dir,
        incremental=False,
        progress_callback=lambda d: events.append(dict(d)),
    )
    kinds = {e["event"] for e in events}
    assert "layout" in kinds
    assert "phase" in kinds
    phase_events = [e for e in events if e.get("event") == "phase"]
    assert phase_events, "expected at least one phase event"
    valid_phase_labels = {"headers", "catalog", "zarr"}
    for e in phase_events:
        assert e["phase"] in valid_phase_labels, f"unknown phase label: {e!r}"
    assert "catalog_row" in kinds
    assert "file_complete" in kinds


def test_scan_from_catalog_for_beamtime_unknown_empty(tmp_path: Path) -> None:
    if getattr(_pyref_mod, "py_scan_from_catalog_for_beamtime", None) is None:
        pytest.skip("py_scan_from_catalog_for_beamtime not built")
    _ensure_flat_beamtime_layout(tmp_path)
    ingest_beamtime(tmp_path, incremental=True)
    db = resolve_catalog_path()
    missing = tmp_path / "not_registered_beamtime"
    df = scan_from_catalog_for_beamtime(missing, db)
    assert df.height == 0
    assert "file_path" in df.columns


def test_list_beamtimes_and_read_beamtime(minimal_fits_dir: Path | None) -> None:
    if getattr(_pyref_mod, "py_scan_from_catalog_for_beamtime", None) is None:
        pytest.skip("py_scan_from_catalog_for_beamtime not built")
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    db = ingest_beamtime(minimal_fits_dir, incremental=False)
    bts = list_beamtimes(db)
    assert bts.height >= 1
    assert "beamtime_path" in bts.columns
    view = read_beamtime(minimal_fits_dir, catalog_path=db, ingest=False)
    assert view.frames.height >= 1
    assert len(view.entries.samples) >= 1


def test_read_beamtime_ingest_quiet(minimal_fits_dir: Path | None) -> None:
    if getattr(_pyref_mod, "py_scan_from_catalog_for_beamtime", None) is None:
        pytest.skip("py_scan_from_catalog_for_beamtime not built")
    if minimal_fits_dir is None:
        pytest.skip("fixtures/minimal.fits not found")
    db = resolve_catalog_path()
    view = read_beamtime(
        minimal_fits_dir,
        catalog_path=db,
        ingest=True,
        show_progress=False,
    )
    assert view.frames.height >= 1
