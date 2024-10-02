"""
Simple interfacing module for reducing reflectometry data.
"""

import pyref_rs as rs
import polars as pl
import numpy as np
from typing import Generator
from masking import InteractiveImageMasker


class Loader:
    """
    Loader class to load RSoXR data from beamline 11.0.1.2 at the ALS

    Parameters
    ----------
    path : str
        Path to the experimental directory

    Attributes
    ----------
    path: str
        Path to the directory
    name: str
        Name of the sample - collected from the directory name

    mask: np.ndarray(bool)
        Mask for the data

    dynamic_range: float
        Dynamic range of the detector

    data: pl.DataFrame
        Data loaded from the directory
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = path.split("/")[-1]
        self._data: pl.DataFrame = pl.DataFrame(rs.py_read_experiment(path, "xrr"))
        self._refl: pl.DataFrame | None = None
        self._mask: None | np.ndarray = None
        self.dynamic_range: None | float = None

    def __repr__(self) -> str:
        s = f"Loader(path={self.path})" + "\n"
        s += f"Name: {self.name}" + "\n"
        s += f"Dynamic Range: {self.dynamic_range}" + "\n"
        s += f"Data: {self.data}" + "\n"
        return s

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def data(self) -> pl.DataFrame:
        current_len = len(self._data)
        self._data = rs.py_simple_update(self._data, self.path)

    @property
    def mask(self) -> None | np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    def draw_mask(self):
        masker = InteractiveImageMasker(self.data.to_numpy())
        self.mask = masker.get_mask()

    def img(self, img_col: str) -> Generator[np.ndarray, None, None]:
        """
        Generator to iterate through images in a specified column.

        Parameters
        ----------
        img_col : str
            The name of the column containing image data.

        Yields
        ------
        np.ndarray
            The image data as a numpy array.
        """
        for img in self.data[img_col]:
            shape = int(np.sqrt(len(img)))
            yield rs.py_get_image(img, [shape, shape])

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterator to iterate through images in the "Raw" column.

        Yields
        ------
        np.ndarray
            The image data as a numpy array.
        """
        yield from self.img("Raw")

    # ============/ Initial Data Processing /============
    def spec_reflectance(self):
        """
        Calculate the specular reflectance from the data, adding a new column to the
        DataFrame.
        """
        return self.beam() - self.bg()

    def locate_beam(self):
        """
        Locate the beam position in the data, adding a new column to the DataFrame.
        """
        ...

    def bg(self) -> int:
        """
        Subtract the background from the data, adding a new column to the DataFrame.
        """
        ...

    def beam(self) -> int:
        """
        Subtract the beam from the data, adding a new column to the DataFrame.
        """
        ...

    # ============/ Stitching /============
    @property
    def refl(self) -> pl.DataFrame:
        if self._refl is None:
            self._refl = self._data.iter_rows().map(
                self.spec_reflectance, in_place=False
            )
        return self._refl

    def stitch(self):
        """
        Stitch the data together to form a contiguous dataset.
        """
        ...

    def avg_overlap(self):
        """
        Average the overlap between datasets.
        """
        ...


if __name__ == "__main__":
    loader = Loader("/home/hduva/projects/pyref-ccd/test")
    print(loader)
    for img in loader:
        print(img)
