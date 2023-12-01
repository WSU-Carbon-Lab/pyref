""" 
FitsIO
------
An object for handling the input and output of fits files.

ReflIO
------
An object for handling the input and output of reflectometry data.

@Author: Harlan Heilman
"""

from pathlib import Path
from typing import Final, Union

import numpy as np
import pandas as pd
from astropy.io import fits

HEADER_LIST: Final[list] = [
    "Beamline Energy",
    "Sample Theta",
    "Beam Current",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
]

HEADER_DICT: Final[dict[str, str]] = {
    "Beamline Energy": "Energy",
    "Sample Theta": "Theta",
    "Beam Current": "Current",
    "Higher Order Suppressor": "HOS",
    "EPU Polarization": "POL",
    "EXPOSURE": "Exposure",
    "DATE": "Date",
}


class FitsIO(Path):
    """
    A subclass of pathlib.Path that incorporates methods for working with
    fits files.

    Parameters
    ----------
    Path : Path | str
        A string or path-like object representing a path on the filesystem.
    """

    # Constructors
    # --------------------------------------------------------------
    def __init__(self, path):
        super().__init__(path)

    # Methods
    # --------------------------------------------------------------
    def get_header(
        self, header_values: list[str] = HEADER_LIST, file_name: bool = False
    ) -> dict:
        """
        Method for extracting header values from a fits file.

        Returns
        -------
        dict
            A dictionary of header values.
        """
        with fits.open(self) as hdul:
            meta = hdul[0].header  # type: ignore

        return {
            HEADER_DICT[key]: round(meta[key], 4)
            if isinstance(meta[key], (int, float))
            else meta[key]
            for key in header_values
            if key in meta
        }

    def get_image(self) -> np.ndarray:
        """
        Method for extracting image data from a fits file.

        Returns
        -------
        np.ndarray
            A numpy array of image data.
        """
        with fits.open(self) as hdul:
            image = hdul[2].data  # type: ignore

        return image

    def get_data(
        self, header_values: list[str] = HEADER_LIST, file_name: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Method for extracting header values and image data from a fits file.

        Parameters
        ----------
        header_values : list[str], optional
            List of header data that is needed to be extracted,
            by default HEADER_LIST
        file_name : bool, optional
            Indicator for if the file names should be saves in the array
            , by default False

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            header data and image data
        """
        with fits.open(self) as hdul:
            meta = hdul[0].header  # type: ignore
            image = hdul[2].data  # type: ignore

        header = pd.DataFrame(
            {
                HEADER_DICT[key]: round(meta[key], 4)
                if isinstance(meta[key], (int, float))
                else meta[key]
                for key in header_values
                if key in meta
            },
            index=[0],
        )
        return header, image


class ReflIO(Path):
    """
    A subclass of pathlib.Path that incorporates methods for working with
    reflectometry data stored in many fits files. This overloads the prior
    ReflIO class.

    Parameters
    ----------
    Path : _type_
        _description_
    """

    def __init__(self, path):
        super().__init__(path)

    def get_fits(self):
        """


        Returns
        -------
        _type_
            _description_
        """
        return list(self.glob("*.fits"))

    def get_header(
        self, header_values: list[str] = HEADER_LIST, file_name: bool = False
    ) -> pd.DataFrame:
        meta = []
        for file in self.get_fits():
            meta.append(FitsIO(file).get_header(header_values, file_name))
        return pd.concat(meta)

    def get_image(self) -> np.ndarray:
        image = []
        for file in self.get_fits():
            image.append(FitsIO(file).get_image())
        return np.array(image)

    def get_data(
        self, header_values: list[str] = HEADER_LIST, file_name: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray]:
        meta = []
        image = []
        for file in self.get_fits():
            meta.append(FitsIO(file).get_header(header_values, file_name))
            image.append(FitsIO(file).get_image())
        return pd.concat(meta), np.array(image)
