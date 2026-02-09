"""
FITS filename parsing, discovery, and catalog building for experiment directories.

Pattern: <samplename><tag?><sep?><experiment_5digits>-<frame_5digits>.fits
Sep: optional whitespace, '-', or '_' between name+tag and experiment number.
Tag: if the part before sep has an underscore, last segment is tag, rest is sample_name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import polars as pl

from pyref.io.readers import read_fits, read_fits_metadata

STEM_PATTERN = re.compile(r"^(.+?)[\s\-_]?(\d{5})-(\d{5})$")


@dataclass(frozen=True)
class ParsedFitsName:
    """Parsed components of a FITS filename stem."""

    sample_name: str
    tag: str | None
    experiment_number: int
    frame_number: int
    file_stem: str


def parse_fits_stem(stem: str) -> ParsedFitsName | None:
    """
    Parse a FITS file stem into sample_name, tag, experiment_number, frame_number.

    Parameters
    ----------
    stem : str
        File stem (filename without .fits), e.g. "ZnPc_rt81041-00001".

    Returns
    -------
    ParsedFitsName | None
        Parsed fields, or None if stem does not match the pattern.

    Examples
    --------
    >>> parse_fits_stem("ZnPc_rt81041-00001")
    ParsedFitsName(sample_name='ZnPc', tag='rt', experiment_number=81041, frame_number=1, ...)
    >>> parse_fits_stem("monlayerjune 81041-00007")
    ParsedFitsName(sample_name='monlayerjune', tag=None, experiment_number=81041, frame_number=7, ...)
    """
    stem = stem.strip()
    m = STEM_PATTERN.match(stem)
    if not m:
        return None
    base, exp_str, frame_str = m.group(1), m.group(2), m.group(3)
    base = base.strip()
    experiment_number = int(exp_str)
    frame_number = int(frame_str)
    if "_" in base:
        parts = base.split("_")
        tag = parts[-1]
        sample_name = "_".join(parts[:-1])
    else:
        sample_name = base
        tag = None
    return ParsedFitsName(
        sample_name=sample_name,
        tag=tag,
        experiment_number=experiment_number,
        frame_number=frame_number,
        file_stem=stem,
    )


def discover_fits(root: Path, recursive: bool = True) -> list[Path]:
    """
    Discover FITS files under a root path.

    Parameters
    ----------
    root : Path
        Directory to search.
    recursive : bool, optional
        If True, search recursively (rglob); else only direct children (glob).
        Default True.

    Returns
    -------
    list[Path]
        Sorted list of absolute paths to .fits files.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    if recursive:
        paths = sorted(root.rglob("*.fits"))
    else:
        paths = sorted(root.glob("*.fits"))
    return [p.resolve() for p in paths]


def _catalog_from_paths(paths: list[Path]) -> pl.DataFrame:
    path_strs: list[str] = []
    file_stems: list[str] = []
    sample_names: list[str] = []
    tags: list[str | None] = []
    exp_nums: list[int] = []
    frame_nums: list[int] = []
    for p in paths:
        stem = p.stem
        parsed = parse_fits_stem(stem)
        path_strs.append(str(p))
        if parsed is None:
            file_stems.append(stem)
            sample_names.append("")
            tags.append(None)
            exp_nums.append(0)
            frame_nums.append(0)
        else:
            file_stems.append(parsed.file_stem)
            sample_names.append(parsed.sample_name)
            tags.append(parsed.tag)
            exp_nums.append(parsed.experiment_number)
            frame_nums.append(parsed.frame_number)
    return pl.DataFrame({
        "path": path_strs,
        "file_stem": file_stems,
        "sample_name": sample_names,
        "tag": tags,
        "experiment_number": exp_nums,
        "frame_number": frame_nums,
    })


def build_catalog(
    paths: list[Path] | Path,
    headers: list[str] | None = None,
    *,
    recursive: bool = True,
) -> pl.DataFrame:
    """
    Build a per-file catalog DataFrame from paths or a directory.

    Parameters
    ----------
    paths : list[Path] | Path
        Either a directory to discover FITS in, or a list of FITS paths.
    headers : list[str] | None, optional
        If provided, read these FITS headers (and Q if Beamline Energy + Sample Theta
        exist) and add columns to the catalog. If None, catalog is names-only (no I/O).
    recursive : bool, optional
        When paths is a directory, whether to discover recursively. Default True.

    Returns
    -------
    pl.DataFrame
        Catalog with columns path, file_stem, sample_name, tag, experiment_number,
        frame_number. If headers were requested, also includes those columns and Q
        when applicable.
    """
    if isinstance(paths, Path):
        path_obj = Path(paths).resolve()
        if not path_obj.is_dir():
            return pl.DataFrame()
        path_list = discover_fits(path_obj, recursive=recursive)
    else:
        path_list = [Path(p).resolve() for p in paths]
    if not path_list:
        return pl.DataFrame()
    catalog = _catalog_from_paths(path_list)
    if headers is None:
        return catalog
    header_keys = list(headers)
    if "Beamline Energy" not in header_keys:
        header_keys.append("Beamline Energy")
    if "Sample Theta" not in header_keys:
        header_keys.append("Sample Theta")
    path_strs = [str(p) for p in path_list]
    try:
        meta = read_fits_metadata(path_strs, headers=header_keys, engine="polars")
    except Exception:
        return catalog
    if not isinstance(meta, pl.DataFrame) or "file_name" not in meta.columns:
        return catalog
    if meta.height != len(path_list):
        return catalog
    catalog = catalog.join(meta, left_on="file_stem", right_on="file_name", how="left")
    if "file_name" in catalog.columns:
        catalog = catalog.drop("file_name")
    return catalog.sort(["experiment_number", "frame_number"])


def scan_view(catalog: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate catalog into a per-scan view: sample_name, tag, experiment_number,
    file_count, and optionally energy_min/max, Q_min/max.

    Parameters
    ----------
    catalog : pl.DataFrame
        Catalog from build_catalog (with or without header columns).

    Returns
    -------
    pl.DataFrame
        One row per (sample_name, tag, experiment_number) with file_count and
        optional energy/Q aggregates.
    """
    group_cols = ["sample_name", "tag", "experiment_number"]
    aggs = [pl.len().alias("file_count")]
    if "Beamline Energy" in catalog.columns:
        aggs.extend([
            pl.col("Beamline Energy").min().alias("energy_min"),
            pl.col("Beamline Energy").max().alias("energy_max"),
        ])
    if "Q" in catalog.columns:
        aggs.extend([
            pl.col("Q").min().alias("Q_min"),
            pl.col("Q").max().alias("Q_max"),
        ])
    aggs.append(pl.col("path").first().alias("path"))
    return catalog.group_by(group_cols).agg(aggs).sort(group_cols)


def experiment_summary(
    root: Path,
    recursive: bool = True,
    with_headers: bool = False,
    headers: list[str] | None = None,
) -> pl.DataFrame:
    """
    Quick view of an experiment directory: per-scan summary table.

    Parameters
    ----------
    root : Path
        Experiment root directory.
    recursive : bool, optional
        Discover FITS recursively. Default True.
    with_headers : bool, optional
        If True, read minimal FITS headers to include energy and Q range in summary.
        Default False.
    headers : list[str] | None, optional
        Header keys when with_headers is True. Defaults to Beamline Energy, Sample Theta, DATE.

    Returns
    -------
    pl.DataFrame
        scan_view of the catalog (sample_name, tag, experiment_number, file_count,
        energy_min/max, Q_min/max when with_headers).
    """
    if with_headers and headers is None:
        headers = ["Beamline Energy", "Sample Theta", "DATE"]
    catalog = build_catalog(Path(root), headers=headers if with_headers else None, recursive=recursive)
    if catalog.is_empty():
        return pl.DataFrame()
    return scan_view(catalog)


def filter_catalog_paths(
    catalog: pl.DataFrame,
    *,
    sample_name: str | None = None,
    tag: str | None = None,
    experiment_number: int | None = None,
    experiment_numbers: list[int] | None = None,
) -> list[Path]:
    """
    Filter catalog by sample_name, tag, or experiment number(s); return list of paths.

    Parameters
    ----------
    catalog : pl.DataFrame
        Catalog from build_catalog.
    sample_name : str | None, optional
        Filter to this sample_name.
    tag : str | None, optional
        Filter to this tag.
    experiment_number : int | None, optional
        Filter to this single experiment number.
    experiment_numbers : list[int] | None, optional
        Filter to any of these experiment numbers.

    Returns
    -------
    list[Path]
        Paths of matching rows.
    """
    df = catalog
    if sample_name is not None:
        df = df.filter(pl.col("sample_name") == sample_name)
    if tag is not None:
        df = df.filter(pl.col("tag") == tag)
    if experiment_number is not None:
        df = df.filter(pl.col("experiment_number") == experiment_number)
    if experiment_numbers is not None:
        df = df.filter(pl.col("experiment_number").is_in(experiment_numbers))
    return [Path(p) for p in df.get_column("path").to_list()]
