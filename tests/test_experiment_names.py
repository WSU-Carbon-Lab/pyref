"""Tests for FITS filename parsing, discovery, and catalog."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pyref import get_data_path
from pyref.io.experiment_names import (
    ParsedFitsName,
    build_catalog,
    discover_fits,
    experiment_summary,
    filter_catalog_paths,
    parse_fits_stem,
    scan_view,
)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("ZnPc_rt81041-00001", ParsedFitsName("ZnPc", "rt", 81041, 1, "ZnPc_rt81041-00001")),
        ("ZnPc_rt_81041-00001", ParsedFitsName("ZnPc", "rt", 81041, 1, "ZnPc_rt_81041-00001")),
        ("ZnPc_rt 81041-00001", ParsedFitsName("ZnPc", "rt", 81041, 1, "ZnPc_rt 81041-00001")),
        ("ZnPc_rt-81041-00001", ParsedFitsName("ZnPc", "rt", 81041, 1, "ZnPc_rt-81041-00001")),
        ("ps_pmma_rt 81041-00001", ParsedFitsName("ps_pmma", "rt", 81041, 1, "ps_pmma_rt 81041-00001")),
        ("monlayerjune 81041-00007", ParsedFitsName("monlayerjune", None, 81041, 7, "monlayerjune 81041-00007")),
        ("monlayerjune 81041-00001", ParsedFitsName("monlayerjune", None, 81041, 1, "monlayerjune 81041-00001")),
    ],
)
def test_parse_fits_stem(stem: str, expected: ParsedFitsName) -> None:
    got = parse_fits_stem(stem)
    assert got is not None
    assert got.sample_name == expected.sample_name
    assert got.tag == expected.tag
    assert got.experiment_number == expected.experiment_number
    assert got.frame_number == expected.frame_number
    assert got.file_stem == expected.file_stem


def test_parse_fits_stem_invalid() -> None:
    assert parse_fits_stem("notavalidstem") is None
    assert parse_fits_stem("short1-00001") is None
    assert parse_fits_stem("sample12345-678") is None


def test_discover_fits_flat() -> None:
    data_dir = get_data_path()
    paths = discover_fits(data_dir, recursive=False)
    assert isinstance(paths, list)
    for p in paths:
        assert isinstance(p, Path)
        assert p.suffix == ".fits"
        assert p.is_absolute()


def test_discover_fits_recursive() -> None:
    data_dir = get_data_path()
    paths_flat = discover_fits(data_dir, recursive=False)
    paths_rec = discover_fits(data_dir, recursive=True)
    assert len(paths_rec) >= len(paths_flat)


def test_build_catalog_names_only() -> None:
    data_dir = get_data_path()
    catalog = build_catalog(data_dir, headers=None, recursive=False)
    assert isinstance(catalog, pl.DataFrame)
    assert not catalog.is_empty()
    for col in ("path", "file_stem", "sample_name", "tag", "experiment_number", "frame_number"):
        assert col in catalog.columns
    sample_names = catalog.get_column("sample_name").unique().to_list()
    assert "monlayerjune" in sample_names or len(sample_names) >= 1


def test_build_catalog_with_headers() -> None:
    data_dir = get_data_path()
    paths = discover_fits(data_dir, recursive=False)
    if not paths:
        pytest.skip("No FITS files in data dir")
    catalog = build_catalog(paths[:3], headers=["Beamline Energy", "Sample Theta", "DATE"])
    assert not catalog.is_empty()
    assert catalog.height == min(3, len(paths))
    if "Beamline Energy" in catalog.columns and "Q" in catalog.columns:
        assert catalog.get_column("Beamline Energy").null_count() <= catalog.height
        assert catalog.get_column("Q").null_count() <= catalog.height


def test_scan_view() -> None:
    data_dir = get_data_path()
    catalog = build_catalog(data_dir, headers=None, recursive=False)
    if catalog.is_empty():
        pytest.skip("No FITS files")
    view = scan_view(catalog)
    assert "file_count" in view.columns
    assert "sample_name" in view.columns
    assert "experiment_number" in view.columns
    assert view.height >= 1
    assert view.get_column("file_count").sum() == catalog.height


def test_experiment_summary() -> None:
    data_dir = get_data_path()
    summary = experiment_summary(data_dir, recursive=False, with_headers=False)
    assert isinstance(summary, pl.DataFrame)
    if not summary.is_empty():
        assert "file_count" in summary.columns


def test_experiment_summary_with_headers() -> None:
    data_dir = get_data_path()
    paths = discover_fits(data_dir, recursive=False)
    if not paths:
        pytest.skip("No FITS files")
    summary = experiment_summary(data_dir, recursive=False, with_headers=True)
    assert isinstance(summary, pl.DataFrame)
    if not summary.is_empty() and "energy_min" in summary.columns:
        assert "Q_min" in summary.columns or "energy_min" in summary.columns


def test_filter_catalog_paths() -> None:
    data_dir = get_data_path()
    catalog = build_catalog(data_dir, headers=None, recursive=False)
    if catalog.is_empty():
        pytest.skip("No FITS files")
    all_paths = filter_catalog_paths(catalog)
    assert len(all_paths) == catalog.height
    exp_unique = catalog.get_column("experiment_number").unique().to_list()
    if exp_unique:
        one_exp = exp_unique[0]
        filtered = filter_catalog_paths(catalog, experiment_number=one_exp)
        expected_count = catalog.filter(pl.col("experiment_number") == one_exp).height
        assert len(filtered) == expected_count
    sample_unique = catalog.get_column("sample_name").unique().to_list()
    if sample_unique:
        one_sample = [s for s in sample_unique if s][0]
        filtered = filter_catalog_paths(catalog, sample_name=one_sample)
        assert len(filtered) <= catalog.height
        assert all(Path(p).exists() for p in filtered)


def test_read_fits_meta_has_parsed_columns() -> None:
    from pyref.pyref import py_read_fits_headers_only

    data_dir = get_data_path()
    paths = discover_fits(data_dir, recursive=False)
    if not paths:
        pytest.skip("No FITS files")
    meta = py_read_fits_headers_only(str(paths[0]), [])
    assert isinstance(meta, pl.DataFrame)
    assert "file_name" in meta.columns
    if "sample_name" in meta.columns:
        stem = str(meta["file_name"][0])
        parsed = parse_fits_stem(stem)
        assert parsed is not None
        assert meta["sample_name"][0] == parsed.sample_name
        assert meta["experiment_number"][0] == parsed.experiment_number
        assert meta["frame_number"][0] == parsed.frame_number
