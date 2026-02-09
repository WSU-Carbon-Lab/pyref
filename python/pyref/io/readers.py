"""
Module contains tools for processing files into DataFrames or other objects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pyref.pyref import (  # type: ignore[import]
    py_read_experiment,
    py_read_experiment_metadata,
    py_read_experiment_pattern,
    py_read_fits,
    py_read_fits_metadata,
    py_read_multiple_fits,
    py_read_multiple_fits_metadata,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

type FilePath = str | Path
type FileDirectory = str | Path
type FilePathList = list[str] | list[Path]
type RegexPattern = str | re.Pattern[str]


def _stem_matches(path: Path, pattern: RegexPattern) -> bool:
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return compiled.search(path.stem) is not None


def read_fits(
    file_path: FilePath,
    headers: list[str] | None = None,
    *,
    pattern: RegexPattern | None = None,
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Read data from a FITS file into a DataFrame.

    Parameters
    ----------
    file_path : str | Path | list[str] | list[Path] | FilePath | FilePathList
        Path to the FITS file, or a list of paths to FITS files to read.
    headers : list[str] | None, optional
        List of header values to parse from the header; use `None` to read all header
        values, by default None
    pattern : str | re.Pattern[str] | None, optional
        Regex applied to file stem (filename without .fits). Only paths whose stem
        matches are read. E.g. ``r"^znpc"`` or ``r"^ZnPc"`` to match stems starting
        with znpc. Ignored when None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the FITS file content.

    Raises
    ------
    FileNotFoundError
        If the file_path does not point to a valid file.
    ValueError
        If the file is not a FITS file (does not end with .fits), or if pattern
        is set and no paths match.

    Notes
    -----
    The constructed DataFrame will always have the following columns:
    - `DATE`: (pl.String) Date time string of when the file was created.
    - `raw`: (pl.Array(pl.Uint64, N, M)) Raw CCD camera image data as a 2D array.

    Example
    -------
    This example shows how to read the FITS files from a directory with a specific
    series of headers:
    >>> from pyref.io import read_fits
    >>> df = read_fits(
    ...     "path/to/file.fits",
    ...     headers=["DATE", "Beamline Energy", "EXPOSURE"],
    ... )
    >>> print(df)
    Alternatively, you can read all the header values by setting `headers` to `None`:
    >>> df = read_fits("path/to/file.fits", headers=None)
    >>> print(df)
    And alternatively, you can read a list of FITS files:
    >>> df = read_fits(
    ...     ["path/to/file1.fits", "path/to/file2.fits"],
    ...     headers=["DATE", "Beamline Energy", "EXPOSURE"],
    ... )
    >>> print(df)
    Filter by regex on file stem (e.g. only stems starting with znpc):
    >>> df = read_fits(path_list, headers=[...], pattern=r"^znpc")
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
        polars_data = py_read_multiple_fits(file_paths_str, headers)
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
        polars_data = py_read_fits(str(file_path_obj), headers)

    if engine == "pandas":
        return polars_data.to_pandas()
    elif engine == "polars":
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
    Read data from a FITS file or pattern into a DataFrame.

    Parameters
    ----------
    file_path : str | Path | FileDirectory
        Path to the directory containing FITS files.
    headers : list[str] | None, optional
        List of header values to parse from the header use `None` to read all header
        values, by default None
    pattern : str | None, optional
        Glob pattern to match filenames (e.g. ``"*85684*"``). Passed to the backend
        when regex is not set. By default None (all *.fits in directory).
    regex : str | re.Pattern[str] | None, optional
        Regex applied to file stem. Only files whose stem matches are read.
        E.g. ``r"^znpc"`` or ``r"^ZnPc"``. When set, discovery uses recursive
        if recursive is True. By default None.
    recursive : bool, optional
        When regex is set, if True search recursively under file_path (rglob);
        otherwise only direct children (glob). Ignored when regex is None.
        By default False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the FITS file content.

    Raises
    ------
    FileNotFoundError
        If the file_path does not point to a valid directory.
    ValueError
        If the file is not a FITS file (does not end with .fits), or if regex
        is set and no files match.

    Notes
    -----
    The constructed DataFrame will always have the following columns:
    - `DATE`: (pl.String) Date time string of when the file was created.
    - `raw`: (pl.Array(pl.Uint64, N, M)) Raw CCD camera image data as a 2D array.

    Example
    -------
    This example shows how to read the FITS files from a directory with a specific
    series of headers:
    >>> from pyref.io import read_experiment
    >>> df = read_experiment(
    ...     "path/to/directory",
    ...     headers=["DATE", "Beamline Energy", "EXPOSURE"],
    ... )
    >>> print(df)
    Alternatively, you can read all the header values by setting `headers` to `None`:
    >>> df = read_experiment("path/to/directory", headers=None)
    >>> print(df)
    And alternatively, you can read a specific glob pattern of files in the directory:
    >>> df = read_experiment("path/to/directory", pattern="*85684*")
    >>> print(df)
    Filter by regex on file stem (e.g. stems starting with znpc), optionally recursive:
    >>> df = read_experiment("path/to/directory", headers=[...], regex=r"^znpc", recursive=True)
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.is_dir():
        msg = f"{file_path_obj} is not a valid directory."
        raise FileNotFoundError(msg)
    if headers is None:
        headers = []

    if regex is not None:
        if recursive:
            paths = sorted(file_path_obj.rglob("*.fits"))
        else:
            paths = sorted(file_path_obj.glob("*.fits"))
        paths = [p for p in paths if _stem_matches(p, regex)]
        if not paths:
            msg = "No FITS files match the given regex."
            raise ValueError(msg)
        polars_data = py_read_multiple_fits([str(p) for p in paths], headers)
    elif pattern:
        polars_data = py_read_experiment_pattern(str(file_path_obj), pattern, headers)
    else:
        if not any(file_path_obj.glob("*.fits")):
            msg = f"{file_path_obj} does not contain any FITS files."
            raise FileNotFoundError(msg)
        polars_data = py_read_experiment(str(file_path_obj), headers)

    if engine == "pandas":
        return polars_data.to_pandas()
    elif engine == "polars":
        return polars_data


def read_fits_metadata(
    file_path: FilePath | FilePathList,
    headers: list[str] | None = None,
    *,
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Read FITS metadata (headers + file_name + NAXIS1/NAXIS2) without loading image data.
    Use this for catalogs when files may have different image dimensions.
    """
    if headers is None:
        headers = []
    if isinstance(file_path, list):
        path_strs = [str(Path(p).resolve()) for p in file_path]
        for p in path_strs:
            if not Path(p).is_file():
                raise FileNotFoundError(f"{p} is not a valid file.")
        out = py_read_multiple_fits_metadata(path_strs, headers)
    else:
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not a valid file.")
        if p.suffix != ".fits":
            raise ValueError(f"{p} is not a FITS file.")
        out = py_read_fits_metadata(str(p), headers)
    if engine == "pandas":
        return out.to_pandas()
    return out


def read_experiment_metadata(
    file_path: FileDirectory,
    headers: list[str] | None = None,
    *,
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Read metadata for all FITS files in a directory (no image data).
    Use for catalogs when files may have different image dimensions.
    """
    if headers is None:
        headers = []
    path_obj = Path(file_path).resolve()
    if not path_obj.is_dir():
        raise FileNotFoundError(f"{path_obj} is not a valid directory.")
    paths = sorted(path_obj.glob("*.fits"))
    if not paths:
        raise FileNotFoundError(f"{path_obj} does not contain any FITS files.")
    path_strs = [str(p) for p in paths]
    out = py_read_experiment_metadata(str(path_obj), headers)
    if engine == "pandas":
        return out.to_pandas()
    return out
