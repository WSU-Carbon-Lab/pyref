"""
Module contains tools for processing files into DataFrames or other objects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from pyref.io.catalog_path import NEW_CATALOG_DB_NAME, resolve_catalog_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd

type FilePath = str | Path
type FileDirectory = str | Path
type FilePathList = list[str] | list[Path]
type RegexPattern = str | re.Pattern[str]

DEFAULT_HEADER_KEYS = [
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
]

REQUIRED_SCAN_COLUMNS = (
    "file_path",
    "data_offset",
    "naxis1",
    "naxis2",
    "bitpix",
    "bzero",
    "file_name",
)


def _stem_matches(path: Path, pattern: RegexPattern) -> bool:
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return compiled.search(path.stem) is not None


def _is_skippable_stem(stem: str) -> bool:
    return stem.startswith("_") or stem == ""


def resolve_fits_paths(source: FilePath | FilePathList) -> list[str]:
    """
    Normalize ``source`` into a sorted list of absolute ``.fits`` paths.

    Parameters
    ----------
    source : str, pathlib.Path, list, or tuple
        A single FITS file, directory (recursive ``.fits`` discovery), glob-like path,
        or a list/tuple of such paths.

    Returns
    -------
    list of str
        Unique absolute paths, sorted lexicographically.
    """
    if isinstance(source, list | tuple):
        out: list[str] = []
        for p in source:
            path = Path(p).resolve()
            if path.is_file() and path.suffix.lower() == ".fits":
                if not _is_skippable_stem(path.stem):
                    out.append(str(path))
            elif path.is_dir():
                for f in sorted(path.rglob("*.fits")):
                    if f.is_file() and not _is_skippable_stem(f.stem):
                        out.append(str(f.resolve()))
        return sorted(set(out))
    path = Path(source).resolve()
    if path.is_file():
        if path.suffix.lower() != ".fits" or _is_skippable_stem(path.stem):
            return []
        return [str(path)]
    if path.is_dir():
        return sorted(
            str(f.resolve())
            for f in path.rglob("*.fits")
            if f.is_file() and not _is_skippable_stem(f.stem)
        )
    if "*" in path.name or "?" in path.name:
        parent = path.parent
        if not parent.exists():
            return []
        return sorted(
            str(f.resolve())
            for f in parent.glob(path.name)
            if f.is_file()
            and f.suffix.lower() == ".fits"
            and not _is_skippable_stem(f.stem)
        )
    return []


def _scan_schema(header_items: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "file_path": pl.String,
        "data_offset": pl.Int64,
        "naxis1": pl.Int64,
        "naxis2": pl.Int64,
        "bitpix": pl.Int64,
        "data_size": pl.Int64,
        "bzero": pl.Int64,
        "file_name": pl.String,
        "sample_name": pl.String,
        "tag": pl.String,
        "scan_number": pl.Int64,
        "frame_number": pl.Int64,
    }
    for key in header_items:
        if key not in schema:
            schema[key] = pl.Float64
    return schema


def scan_experiment(
    source: FilePath | FilePathList,
    header_items: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Lazy Polars metadata from a SQLite catalog or from FITS headers.

    When ``source`` is a **single** path (not a list or tuple):

    - File named ``catalog.db`` (typically under ``.pyref``): load via
      :func:`pyref.pyref.py_scan_from_catalog`.
    - Directory: resolve DB with :func:`pyref.io.catalog_path.resolve_catalog_path`;
      if it exists, load with ``py_scan_from_catalog`` (Rust-aligned layouts).
    - Else: discover ``.fits`` and stream header reads.

    For a **list or tuple** ``source``, call :func:`resolve_fits_paths` only. Pass one
    directory path to use catalog auto-resolution.

    Missing ``py_scan_from_catalog`` falls through to FITS discovery without raising.

    Parameters
    ----------
    source : str, pathlib.Path, or list thereof
        Beamtime directory, catalog ``.db`` file, glob-like path, or FITS path list.
    header_items : list of str, optional
        Extra FITS header keys for disk reads (unused for pure catalog loads).

    Returns
    -------
    polars.LazyFrame
        Schema compatible with :data:`REQUIRED_SCAN_COLUMNS` and catalog scans.
    """
    from pyref.pyref import py_read_multiple_fits_headers_only

    keys = header_items if header_items is not None else []
    if not isinstance(source, list | tuple):
        path = Path(source).resolve()
        if path.is_file() and path.name == NEW_CATALOG_DB_NAME:
            try:
                from pyref.pyref import py_scan_from_catalog

                df = py_scan_from_catalog(str(path), None)
                return df.lazy()
            except (ImportError, AttributeError):
                pass
        if path.is_dir():
            db = resolve_catalog_path(path)
            if db.is_file():
                try:
                    from pyref.pyref import py_scan_from_catalog

                    df = py_scan_from_catalog(str(db), None)
                    return df.lazy()
                except (ImportError, AttributeError):
                    pass
    paths = resolve_fits_paths(source)
    if not paths:
        return pl.DataFrame(schema=pl.Schema(_scan_schema(keys))).lazy()
    schema = _scan_schema(keys)

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ):
        batch_size = batch_size or 50
        total = 0
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            df = py_read_multiple_fits_headers_only(batch, keys)
            df = df.filter(
                (pl.col("sample_name") != "")
                | (pl.col("tag").is_not_null() & (pl.col("tag") != ""))
            )
            if with_columns:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            yield df
            total += len(df)
            if n_rows is not None and total >= n_rows:
                break

    return pl.io.plugins.register_io_source(
        io_source,
        schema=pl.Schema(schema),
        validate_schema=False,
    )


def beamtime_ingest_layout(beamtime_path: FilePath) -> dict[str, Any]:
    """
    Summarize ingest layout per scan without reading FITS pixels.

    Use this before :func:`ingest_beamtime` to size a Rich ``Progress`` display (or
    custom UI): one task with ``total=layout["total_files"]`` and one task per scan
    (see ``progress_callback`` on :func:`ingest_beamtime`).

    Parameters
    ----------
    beamtime_path : str or pathlib.Path
        Beamtime root directory (same layout rules as ingest).

    Returns
    -------
    dict
        ``total_files`` (int) and ``scans`` (list of per-scan dicts with
        ``scan_number`` and ``files``), ordered by ``scan_number``. Unparseable stems
        use scan ``0``.
    """
    from pyref.pyref import py_beamtime_ingest_layout

    return dict(py_beamtime_ingest_layout(str(Path(beamtime_path).resolve())))


def ingest_beamtime(
    beamtime_path: FilePath,
    header_items: list[str] | None = None,
    *,
    incremental: bool = True,
    worker_threads: int | None = None,
    resource_fraction: float | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    max_scans: int | None = None,
    scan_numbers: list[int] | None = None,
) -> Path:
    """
    Ingest a beamtime directory into the global catalog and local zarr cache.

    Parameters
    ----------
    beamtime_path : str or pathlib.Path
        Root directory of the beamtime (ALS layout with ``CCD`` or ``Axis Photonique``).
    header_items : list of str, optional
        FITS primary HDU keys to read during ingest; defaults to the same keys as
        :data:`DEFAULT_HEADER_KEYS` extended by the Rust ingest list when omitted.
    incremental : bool, optional
        Reserved for future incremental ingest semantics; passed through to Rust.
    worker_threads : int, optional
        Parallel FITS reader worker count. Mutually exclusive with
        ``resource_fraction``.
    resource_fraction : float, optional
        Fraction of ``available_parallelism()`` used for reader workers, in ``(0, 1]``.
        Mutually exclusive with ``worker_threads``.
    progress_callback : callable, optional
        Invoked on the thread that runs Rust ingest (the GIL is released during heavy
        work; the callback re-acquires it). Each call receives a mapping with:

        - ``event == "layout"``: ``total_files``, ``scans`` (each with ``scan_number``,
          ``files``).
        - ``event == "phase"``: ``phase`` is ``headers``, ``catalog``, or ``zarr``.
        - ``event == "catalog_row"``: after each file's rows are inserted into SQLite;
          same counter fields as ``file_complete`` (``scan_*``, ``global_*``).
        - ``event == "file_complete"``: after each zarr write; includes ``scan_number``,
          ``scan_done``, ``scan_total``, ``global_done``, ``global_total``.

        For Rich ``Progress``: size each task for ``2 * file_count`` (one step per
        ``catalog_row``, one per ``file_complete``) and call ``update(...,
        advance=1)`` for both event types. Handle ``phase`` events so the description
        updates during long ``headers`` work before ``catalog_row`` events begin.
    max_scans : int, optional
        Ingest only the first ``max_scans`` scan groups in ascending scan-number order.
        Mutually exclusive with ``scan_numbers``.
    scan_numbers : list of int, optional
        Ingest only these stem-derived scan numbers, in the given order.
        Mutually exclusive with ``max_scans``.

    Returns
    -------
    pathlib.Path
        Absolute path to the global ``catalog.db``.
    """
    from pyref.pyref import py_ingest_beamtime

    keys = header_items if header_items is not None else list(DEFAULT_HEADER_KEYS)
    out = py_ingest_beamtime(
        str(Path(beamtime_path).resolve()),
        keys,
        incremental,
        worker_threads,
        resource_fraction,
        progress_callback,
        max_scans,
        scan_numbers,
    )
    return Path(out)


def get_overrides(
    catalog_path: FilePath,
    path: str | None = None,
) -> pl.DataFrame:
    """
    Read user metadata overrides from the ``file_overrides`` table.

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        Path to the global catalog database (for example the return value of
        :func:`ingest_beamtime`).
    path : str, optional
        When given, return only overrides whose ``source_path`` equals this string.
        When omitted, return all override rows.

    Returns
    -------
    polars.DataFrame
        Zero or more rows with columns ``path``, ``sample_name``, ``tag``, and
        ``notes``.
    """
    from pyref.pyref import py_get_overrides

    return py_get_overrides(str(Path(catalog_path).resolve()), path)


def set_override(
    catalog_path: FilePath,
    path: str,
    sample_name: str | None = None,
    tag: str | None = None,
    notes: str | None = None,
) -> None:
    """
    Upsert sample name, tag, or notes for a row in ``file_overrides``.

    The ``path`` must match ``files.nas_uri`` for an ingested file (use the
    ``file_path`` column from :func:`scan_experiment`).

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        Path to the catalog SQLite database.
    path : str
        Must match ``files.nas_uri`` for an existing file row.
    sample_name : str, optional
        Override sample name, or ``None`` to clear that field on upsert.
    tag : str, optional
        Override tag, or ``None`` to clear that field on upsert.
    notes : str, optional
        Free-form notes, or ``None`` to clear that field on upsert.

    Raises
    ------
    RuntimeError
        When ``path`` is not present in either ``files`` or ``bt_scan_points``.
    """
    from pyref.pyref import py_set_override

    py_set_override(
        str(Path(catalog_path).resolve()),
        path,
        sample_name,
        tag,
        notes,
    )


def set_scan_type_for_beamtime_scan(
    catalog_path: FilePath,
    beamtime_path: FilePath,
    scan_number: int,
    scan_type: Literal["fixed_energy", "fixed_angle"],
) -> None:
    """
    Persist scan classification for one scan in one beamtime.

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        Path to the catalog SQLite database.
    beamtime_path : str or pathlib.Path
        Beamtime root directory used to scope the scan lookup.
    scan_number : int
        Scan number within ``beamtime_path``.
    scan_type : {"fixed_energy", "fixed_angle"}
        Classification to store in ``scans.scan_type``.
    """
    from pyref.pyref import py_set_scan_type_for_beamtime_scan

    py_set_scan_type_for_beamtime_scan(
        str(Path(catalog_path).resolve()),
        str(Path(beamtime_path).resolve()),
        int(scan_number),
        scan_type,
    )


def classify_reflectivity_scan_type(
    pairs: list[tuple[float | None, float | None]],
) -> tuple[str, float | None, float | None, float | None, float | None]:
    r"""
    Classify a reflectivity acquisition from (energy eV, sample theta deg) pairs.

    Wraps the Rust catalog classifier used by the TUI. ``scan_kind`` is one of
    ``\"fixed_energy\"`` (theta scan at nearly fixed energy),
    ``\"fixed_angle\"`` (energy scan at nearly fixed theta), or
    ``\"single_point\"``.

    Parameters
    ----------
    pairs : list of (float or None, float or None)
        One tuple per frame or scan point. Either coordinate may be omitted.

    Returns
    -------
    scan_kind : str
        ``fixed_energy``, ``fixed_angle``, or ``single_point``.
    e_min, e_max, t_min, t_max : float or None
        Extrema over finite samples; ``None`` when an axis has no data.
    """
    from pyref.pyref import py_classify_scan_type

    return py_classify_scan_type(pairs)


def query_catalog(
    catalog_path: FilePath,
    *,
    sample_name: str | None = None,
    tag: str | None = None,
    scan_numbers: list[int] | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> pl.DataFrame:
    """
    Query the global catalog for scan rows matching optional filters.

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        Path to the catalog SQLite database.
    sample_name, tag : str, optional
        Exact-match filters when provided.
    scan_numbers : list of int, optional
        Restrict to these scan numbers.
    energy_min, energy_max : float, optional
        Inclusive beamline energy bounds in eV.

    Returns
    -------
    polars.DataFrame
        Rows returned by the Rust ``py_scan_from_catalog`` bridge.
    """
    from pyref.pyref import py_scan_from_catalog

    filt = {}
    if sample_name is not None:
        filt["sample_name"] = sample_name
    if tag is not None:
        filt["tag"] = tag
    if scan_numbers is not None:
        filt["scan_numbers"] = scan_numbers
    if energy_min is not None:
        filt["energy_min"] = energy_min
    if energy_max is not None:
        filt["energy_max"] = energy_max
    df = py_scan_from_catalog(
        str(Path(catalog_path).resolve()),
        filt if filt else None,
    )
    return df


def get_image(meta_df: pl.DataFrame, row_index: int) -> object:
    """Return raw detector pixels for ``row_index`` via the Rust image bridge."""
    from pyref.pyref import py_get_image

    return py_get_image(meta_df, row_index)


def get_image_filtered(meta_df: pl.DataFrame, row_index: int, sigma: float) -> object:
    """Return a Gaussian-filtered image for ``row_index`` (Rust-backed)."""
    from pyref.pyref import py_materialize_image_filtered

    return py_materialize_image_filtered(meta_df, row_index, sigma)


def get_image_corrected(
    meta_df: pl.DataFrame,
    row_index: int,
    bg_rows: int = 10,
    bg_cols: int = 10,
) -> object:
    """
    Return row/column background-subtracted pixels for ``row_index`` (Rust-backed).
    """
    from pyref.pyref import py_get_image_corrected

    return py_get_image_corrected(meta_df, row_index, bg_rows, bg_cols)


def get_image_filtered_edges(
    meta_df: pl.DataFrame,
    row_index: int,
    sigma: float,
    bg_rows: int = 10,
    bg_cols: int = 10,
) -> object:
    """Return filtered-edge-processed pixels for ``row_index`` (Rust-backed)."""
    from pyref.pyref import py_materialize_image_filtered_edges

    return py_materialize_image_filtered_edges(
        meta_df, row_index, sigma, bg_rows, bg_cols
    )


def read_fits(
    file_path: FilePath | FilePathList,
    headers: list[str] | None = None,
    *,
    pattern: RegexPattern | None = None,
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Load full scan metadata into memory (anti-pattern).

    Equivalent to ``scan_experiment(...).collect()``: bypasses lazy optimizations
    (predicate pushdown, projection pushdown, streaming). Prefer
    ``scan_experiment(source)`` with ``.filter()``, ``.select()``, then
    ``.collect()`` when possible; use this only when the entire result must be
    materialized (e.g. small directories or legacy scripts).
    """
    if isinstance(file_path, list):
        file_paths_str = []
        for fp in file_path:
            p = Path(fp)
            if not p.is_file():
                msg = f"{p} is not a valid file."
                raise FileNotFoundError(msg)
            if p.suffix != ".fits":
                msg = f"{p} is not a FITS file."
                raise ValueError(msg)
            if pattern is None or _stem_matches(p, pattern):
                file_paths_str.append(str(p))
        if pattern is not None and not file_paths_str:
            msg = "No paths match the given pattern."
            raise ValueError(msg)
        source = file_paths_str
    else:
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            msg = f"{file_path_obj} is not a valid file."
            raise FileNotFoundError(msg)
        if file_path_obj.suffix != ".fits":
            msg = f"{file_path_obj} is not a FITS file."
            raise ValueError(msg)
        if pattern is not None and not _stem_matches(file_path_obj, pattern):
            msg = "File stem does not match the given pattern."
            raise ValueError(msg)
        source = file_path_obj
    header_list = headers if headers is not None else []
    polars_data = cast(
        "pl.DataFrame",
        scan_experiment(source, header_items=header_list).collect(),
    )
    if engine == "pandas":
        return polars_data.to_pandas()
    return polars_data


def read_experiment(
    file_path: FileDirectory,
    headers: list[str] | None = None,
    pattern: str | None = None,
    *,
    regex: RegexPattern | None = None,
    recursive: bool = False,
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Load a directory of FITS into memory (anti-pattern).

    Equivalent to ``scan_experiment(...).collect()`` without lazy filtering.
    Prefer ``scan_experiment`` with ``.filter()``, ``.select()``, and
    ``.collect()`` unless the full result is explicitly required.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.is_dir():
        msg = f"{file_path_obj} is not a valid directory."
        raise FileNotFoundError(msg)
    header_list = headers if headers is not None else []
    if regex is not None:
        if recursive:
            paths = sorted(file_path_obj.rglob("*.fits"))
        else:
            paths = sorted(file_path_obj.glob("*.fits"))
        paths = [p for p in paths if _stem_matches(p, regex)]
        if not paths:
            msg = "No FITS files match the given regex."
            raise ValueError(msg)
        source = [str(p) for p in paths]
    elif pattern:
        paths = sorted(file_path_obj.glob(pattern))
        paths = [p for p in paths if p.is_file() and p.suffix.lower() == ".fits"]
        if not paths:
            msg = "No FITS files match the given pattern."
            raise ValueError(msg)
        source = [str(p) for p in paths]
    else:
        if not any(file_path_obj.glob("*.fits")):
            msg = f"{file_path_obj} does not contain any FITS files."
            raise FileNotFoundError(msg)
        source = file_path_obj
    polars_data = cast(
        "pl.DataFrame",
        scan_experiment(source, header_items=header_list).collect(),
    )
    if engine == "pandas":
        return polars_data.to_pandas()
    return polars_data
