"""
Simple interfacing module for reducing reflectometry data.
"""

import pyref_rs as rs
import polars as pl
import numpy as np


class Loader:
    """
    Loader class to load RSoXR data from beamline 11.0.1.2 at the ALS

    Parameters
    ----------
    path : str
        Path to the experimental directory

    Attributes
    ----------
    name: str
        Name of the sample - collected from the directory name

    mask: np.ndarray(bool)
        Mask for the data

    dynamic_range: float
        Dynamic range of the detector

    filter_strength: float
        Filter strength of the data

    data: pl.DataFrame
        Data loaded from the directory
    """

    def __init__(self, path: str) -> None:
        self.name = path.split("/")[-1]
        self.data: pl.DataFrame = pl.DataFrame(rs.py_read_experiment(path, "xrr"))
        self.mask: None | np.ndarray = None
        self.dynamic_range: None | float = None


if __name__ == "__main__":
    loader = Loader("/home/hduva/projects/pyref-ccd/test")
    print(loader.data)
