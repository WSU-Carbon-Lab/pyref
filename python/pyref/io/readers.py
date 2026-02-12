"""
Module contains tools for processing files into DataFrames or other objects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
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


def resolve_fits_paths(source: FilePath | FilePathList) -> list[str]:
    if isinstance(source, (list, tuple)):
        out: list[str] = []
        for p in source:
            path = Path(p).resolve()
            if path.is_file() and path.suffix.lower() == ".fits":
                out.append(str(path))
            elif path.is_dir():
                for f in sorted(path.rglob("*.fits")):
                    if f.is_file():
                        out.append(str(f.resolve()))
        return sorted(set(out))
    path = Path(source).resolve()
    if path.is_file():
        return [str(path)] if path.suffix.lower() == ".fits" else []
    if path.is_dir():
        return sorted(str(f.resolve()) for f in path.rglob("*.fits") if f.is_file())
    if "*" in path.name or "?" in path.name:
        parent = path.parent
        if not parent.exists():
            return []
        return sorted(str(f.resolve()) for f in parent.glob(path.name) if f.is_file() and f.suffix.lower() == ".fits")
    return []


def _scan_schema(header_items: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "file_path": pl.String,
        "data_offset": pl.Int64,
        "naxis1": pl.Int64,
        "naxis2": pl.Int64,
        "bitpix": pl.Int64,
        "bzero": pl.Int64,
        "file_name": pl.String,
        "sample_name": pl.String,
        "tag": pl.String,
        "experiment_number": pl.Int64,
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
    from pyref.pyref import py_read_multiple_fits_headers_only

    keys = header_items if header_items is not None else []
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


def get_image(meta_df: pl.DataFrame, row_index: int) -> object:
    from pyref.pyref import py_get_image

    return py_get_image(meta_df, row_index)


def get_image_filtered(meta_df: pl.DataFrame, row_index: int, sigma: float) -> object:
    from pyref.pyref import py_materialize_image_filtered

    return py_materialize_image_filtered(meta_df, row_index, sigma)


def get_image_corrected(
    meta_df: pl.DataFrame,
    row_index: int,
    bg_rows: int = 10,
    bg_cols: int = 10,
) -> object:
    from pyref.pyref import py_get_image_corrected

    return py_get_image_corrected(meta_df, row_index, bg_rows, bg_cols)


def get_image_filtered_edges(
    meta_df: pl.DataFrame,
    row_index: int,
    sigma: float,
    bg_rows: int = 10,
    bg_cols: int = 10,
) -> object:
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
    Anti-pattern: Equivalent to scan_experiment(...).collect(). Loads the full
    scan result into memory and bypasses lazy optimizations (predicate pushdown,
    projection pushdown, streaming). Prefer scan_experiment(source) with
    .filter(), .select(), then .collect() only when needed; use this only when
    you explicitly need the entire result in memory (e.g. small dirs or legacy scripts).
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
    polars_data = scan_experiment(source, header_items=header_list).collect()
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
    Anti-pattern: Equivalent to scan_experiment(...).collect(). Loads the full
    scan result into memory and bypasses lazy optimizations. Prefer
    scan_experiment(source) with .filter(), .select(), and .collect() only when
    needed; use this only when the full result is explicitly required.
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
    polars_data = scan_experiment(source, header_items=header_list).collect()
    if engine == "pandas":
        return polars_data.to_pandas()
    return polars_data
