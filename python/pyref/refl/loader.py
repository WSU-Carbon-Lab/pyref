"""Simple interfacing module for reducing reflectometry data."""

from collections.abc import Generator

import numpy as np
import polars as pl
import pyref_rs as rs
from result import Ok

from pyref.masking import InteractiveImageMasker

# ============/ Light Rust Wrappers /============


class Loader:
    """
    Loader class to load RSoXR data from beamline 11.0.1.2 at the ALS.

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
        self.raw: pl.DataFrame = Ok(rs.py_read_experiment(path, "xrr")).unwrap_err(
            "Failed to read experiment"
        )

    def __repr__(self) -> str:
        return self.raw.__repr__()

    def __str__(self) -> str:
        return self.raw.__str__()

    @property
    def raw(self) -> pl.DataFrame:
        self.raw = rs.py_simple_update(self.raw, self.path)

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
        Image iterator.

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
        Raw Image Iterator.

        Yields
        ------
        np.ndarray
            The image data as a numpy array.
        """
        yield from self.img("Raw")

    # ============/ Initial Data Processing /============
    def spec_reflectance(self):
        """Calculate the specular reflectance from the data."""
        return self.beam() - self.bg()

    def locate_beam(self):
        """
        Locate the beam position in the data, adding a new column to the DataFrame.
        """

    def bg(self) -> int:
        """
        Subtract the background from the data, adding a new column to the DataFrame.
        """

    def beam(self) -> int:
        """
        Subtract the beam from the data, adding a new column to the DataFrame.
        """

    # ============/ Stitching /============
    def stitch(self):
        """
        Stitch the data together to form a contiguous dataset.
        """

    def avg_overlap(self):
        """
        Average the overlap between datasets.
        """


if __name__ == "__main__":
    loader = Loader("/home/hduva/projects/pyref-ccd/test")
    print(loader)
    print(loader.raw)
