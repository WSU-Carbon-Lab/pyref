"""
ReflDataFrame
-------------
A subclass of pandas.DataFrame that contains 2d Data, metadata, and 
associated methods for working with reflectometry data.

@Author: Harlan Heilman
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd

from reflutils.core.io import ReflIO

ArrayLike = Union[np.ndarray, list]


class ReflDataFrame(pd.DataFrame):
    """
    A subclass of pandas.DataFrame that contains 2d Data, metadata, and
    associated methods for working with reflectometry data.

    Parameters
    ----------
    pd : _type_
        _description_
    """

    # --------------------------------------------------------------
    # constructors

    def __init__(self, raw_images=None, meta_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(raw_images, ArrayLike):
            self.raw_images = raw_images
            self.meta_data = meta_data

        elif isinstance(raw_images, ReflIO):
            self.raw_images = raw_images.get_image()
            self.meta_data = raw_images.get_header()

    def __repr__(self):
        return super().__repr__() + "\n" + self.meta_data.__repr__()

    # --------------------------------------------------------------
    # properties

    @property
    def raw_images(self):
        return self._raw_images

    @raw_images.setter
    def raw_images(self, value):
        self._raw_images = value

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, value):
        self._meta_data = value

    # --------------------------------------------------------------
    # methods

    def to_npy(self, path):
        """
        Save the raw images to a .npy file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        """
        np.save(path, self.raw_images)

    def to_parquet(self, path):
        """
        Save the dataframe to a .parquet file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        """
        self.to_parquet(path)

    def to_refnx_dataset(self):
        """
        Convert the dataframe to a refnx.DataSet.

        Returns
        -------
        refnx.DataSet
            The refnx.DataSet object.
        """

        try:
            refl = (self["Q"], self["R"], self["dR"])
            return refl
        except KeyError:
            warnings.warn("The dataframe does not contain the required columns.")
            return None
