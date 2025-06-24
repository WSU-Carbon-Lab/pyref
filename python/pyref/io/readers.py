"""
Module contains tools for processing files into DataFrames or other objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pyref.pyref import (  # type: ignore[import]
    py_read_experiment,
    py_read_experiment_pattern,
    py_read_fits,
    py_read_multiple_fits,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

type FilePath = str | Path
type FileDirectory = str | Path
type FilePathList = list[str] | list[Path]


def read_fits(
    file_path: FilePath,
    headers: list[str] | None = None,
    *,
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

    Returns
    -------
    pd.DataFrame
        DataFrame containing the FITS file content.

    Raises
    ------
    FileNotFoundError
        If the file_path does not point to a valid file.
    ValueError
        If the file is not a FITS file (does not end with .fits).

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
    """
    if isinstance(file_path, list):
        # Handle list of file paths
        file_paths_str: list[str] = []
        for fp in file_path:
            file_path_obj = Path(fp)
            if not file_path_obj.is_file():
                msg = f"{file_path_obj} is not a valid file."
                raise FileNotFoundError(msg)
            if file_path_obj.suffix != ".fits":
                msg = f"{file_path_obj} is not a FITS file."
                raise ValueError(msg)
            file_paths_str.append(str(file_path_obj))
        polars_data = py_read_multiple_fits(file_paths_str, headers)

    else:
        # Handle single file path
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            msg = f"{file_path_obj} is not a valid file."
            raise FileNotFoundError(msg)
        if file_path_obj.suffix != ".fits":
            msg = f"{file_path_obj} is not a FITS file."
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
    engine: Literal["pandas", "polars"] = "polars",
) -> pd.DataFrame | pl.DataFrame:
    """
    Read data from a FITS file or pattern into a DataFrame.

    Parameters
    ----------
    file_path : str | Path | FileDirectory
        Path to the FITS files to read.
    headers : list[str] | None, optional
        List of header values to parse from the header use `None` to read all header
        values, by default None
    pattern : str | None, optional
        Pattern to search in directory, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the FITS file content.

    Raises
    ------
    FileNotFoundError
        If the file_path does not point to a valid file.
    ValueError
        If the file is not a FITS file (does not end with .fits).

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
    And alternatively, you can read a specific pattern of files in the directory:
    >>> df = read_experiment("path/to/directory", pattern="*85684*")
    >>> print(df)
    """
    file_path_obj = Path(file_path)
    if not pattern and not file_path_obj.is_dir():
        msg = f"{file_path_obj} is not a valid directory."
        raise FileNotFoundError(msg)
    if headers is None:
        headers = []
    if pattern:
        polars_data = py_read_experiment_pattern(str(file_path_obj), pattern, headers)

    else:
        # Ensure it's at least one FITS file in the directory
        if not any(file_path_obj.glob("*.fits")):
            msg = f"{file_path_obj} does not contain any FITS files."
            raise FileNotFoundError(msg)
        polars_data = py_read_experiment(str(file_path_obj), headers)
    if engine == "pandas":
        return polars_data.to_pandas()
    elif engine == "polars":
        return polars_data
