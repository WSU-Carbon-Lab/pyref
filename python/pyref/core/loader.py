from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from astropy.io import fits

from pyref.core.config import AppConfig as config
from pyref.core.exceptions import FitsReadError


class Fits:
    """A pandas Series object with a FITS file as the data source."""

    def __init__(self, fits_file, **kwargs):
        header, shape, data = self._read_fits(fits_file)
        self._data = header | {"SHAPE": shape} | {"DATA": data}
        print(self._data)

    def __call__(self, **kwds: Any) -> Any:
        """Constructs a pandas Series object from the FITS file data."""
        series = pl.DataFrame(
            data=self._data, schema=config.FITS_HEADER_TYPES, strict=False
        ).T
        return series

    def __repr__(self) -> str:
        return super().__repr__()

    def _image_to_data(self, image: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
        """
        Convert a 2D image to a 1D array of data.

        Parameters
        ----------
        image : np.ndarray
            2D image data

        Returns
        -------
        tuple[tuple[int, int], np.ndarray]
            Shape of the image and the image data as a 1D array
        """
        shape = image.shape
        data = image.flatten()
        return shape, data

    def _read_fits(self, fits_file: str | Path) -> tuple[dict[str, Any], np.ndarray]:
        """
        Main process for read a fits file and extracting its data.

        Parameters
        ----------
        fits_file : str | Path
            Path location of the fits file.

        Returns
        -------
        Header: dict[str, Any], Images: np.ndarray
            Header and image data from the fits file

        Raises
        ------
        FitsReadError
            If the header or image data could not be read from the fits file
        FitsReadError
            If the header or image data could not be read from the fits file
        """
        # open the fits file and extract the header and image data
        with fits.open(fits_file) as hdul:
            _header = getattr(hdul[0], "header", None)
            _image = getattr(hdul[2], "data", None)

        # ensure the fits files are not empty
        if _header is None:
            error = f"Could not read header from {fits_file}"
            raise FitsReadError(error)
        if _image is None:
            error = f"Could not read image from {fits_file}"
            raise FitsReadError(error)

        # convert the header and image data to the correct types
        header = {config.FITS_HEADER[key]: _header[key] for key in config.FITS_HEADER}
        images = np.array(_image)
        shape, data = self._image_to_data(images)
        return header, shape, data

    def __getattr__(self, name) -> Any:
        return super().__getattr__(name)

    def __setattr__(self, name, value: Any) -> None:
        return super().__setattr__(name, value)

    def __dir__(self) -> list:
        return super().__dir__()  # type: ignore

    def __getitem__(self, name) -> Any:
        return super().__getitem__(name)


class FitsDataFrame(pl.DataFrame):
    """A pandas DataFrame extension spescific for handling experiment directories."""

    def __init__(self, experiment_directory, *args, **kwargs) -> None:
        fits_file = [
            Fits(file)()
            for file in Path(experiment_directory).glob("*.fits")
            if file.is_file()
        ]
        self._locate_stitches(fits_file)
        df = pl.concat(fits_file, **kwargs).T
        self.__dict__.update(df.__dict__)

        self.astype(config.FITS_HEADER_TYPES, copy=False)

    def _locate_stitches(self, fits_file: list[Fits]) -> None:
        stitch = 0
        for i in range(len(fits_file)):
            if i == 0 or fits_file[i]["THETA"] == 0:
                stitch = 0
            elif (fits_file[i - 1]["THETA"] == 0) or (
                fits_file[i]["THETA"] < fits_file[i - 1]["THETA"]
            ):
                stitch += 1

            fits_file[i]["STITCH"] = stitch


if __name__ == "__main__":
    import polars as pl

    data_dir = config.DATA_DIR + "ZnPc82862-00001.fits"
    df = Fits(data_dir)
    print(df())
