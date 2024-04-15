from pathlib import Path
from typing import Any

import numpy as np
import polars as pd
from astropy.io import fits

from pyref.core.config import AppConfig as config
from pyref.core.types import DataDirectory, Value
from python.pyref.core.exceptions import FitsReadError, ScanError


class Fits(pd.Series):
    """A pandas Series object with a FITS file as the data source."""

    def __init__(self, fits_file, *args, **kwargs):
        header, images = self._read_fits(fits_file)
        super().__init__(header, *args, **kwargs)  # type: ignore
        self.astype("float64[pyarrow]")
        self.file_name = Path(fits_file).stem
        self.images = images

    def __repr__(self) -> str:
        title = f"FITS file: {self.file_name}\n"
        image = f"Image shape: {self.images.shape}\n"
        header = "Header:\n"
        return title + image + header + super().__repr__()

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
        return header, images

    def __getattr__(self, name) -> Any:
        return super().__getattr__(name)

    def __setattr__(self, name, value: Any) -> None:
        return super().__setattr__(name, value)

    def __dir__(self) -> list:
        return super().__dir__()  # type: ignore

    def __getitem__(self, name) -> Any:
        return super().__getitem__(name)
