"""
Modue contains tools for processing files into DataFrames or other objects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing import TYPE_CHECKING
from astropy.io import fits

if TYPE_CHECKING:
    from typing import Literal

type FilePath = str | Path 


def read_fits(
        file_path: str | Path

        ) -> pd.DataFrame:
    """
    Fits file data extracted as a dictionary.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.

    Returns
    -------
    dict
        Dictionary containing the FITS file content.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"File {file_path} does not exist."
        raise FileNotFoundError(msg)
    with fits.open(str(file_path)) as hdul:  # type: ignore
        data = pd.DataFrame({hdul[i].name: hdul[i].data for i in range(len(hdul))})

    return data


def read_fits_directory(
    directory: FilePath,
    headers: list[str] =
    calculate_columns: list[str] = None,
) -> pd.DataFrame:
    """
    Directory of FITS files read into a single DataFrame.

    Parameters
    ----------
    directory : str
        Path to the directory containing FITS files.
    file_extension : str
        File extension of the FITS files to read.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames containing the data from each FITS file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        msg = f"{directory} is not a valid directory."
        raise NotADirectoryError(msg)

    fits_files = directory.glob(f"*{file_extension}")
    dataframes = [read_fits(file) for file in fits_files]

    return dataframes
