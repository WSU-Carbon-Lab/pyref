"""
Pandas extensions for handling image data in DataFrames.

This module provides a custom pandas extension type for storing 2D numpy arrays (images)
and an accessor for convenient operations on these image columns.
"""

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import is_array_like


class ImageArray(ExtensionArray):
    """Custom ExtensionArray to store 2D numpy arrays (images)."""

    _itemsize = np.dtype(object).itemsize  # Typical for object arrays

    def __init__(self, values, dtype=None, copy=False):
        if not (isinstance(values, np.ndarray) and values.dtype == object):
            values = np.array(values, dtype=object)

        self._data = values.copy() if copy else values
        self._dtype = dtype or ImageDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ImageArray from a sequence of scalars.

        Each scalar is expected to be a 2D np.ndarray or convertible to one.
        """
        if dtype is None:
            dtype = ImageDtype()

        data = []
        for s in scalars:
            if isinstance(s, np.ndarray) and s.ndim == 2:  # Already a 2D np.ndarray
                data.append(s.copy() if copy else s)
            elif is_array_like(s):  # Try to stack it if it's array-like
                try:
                    arr = np.stack(s)  # type: ignore
                    if arr.ndim != 2:
                        # If not 2D, try converting directly
                        arr = np.array(s)
                        if arr.ndim != 2:
                            msg = f"Cannot convert to 2D array: {type(s)}"
                            raise ValueError(msg)  # noqa: TRY301
                    data.append(arr.copy() if copy else arr)
                except Exception as e:
                    # If stacking fails, try direct conversion
                    try:
                        arr = np.array(s)
                        if arr.ndim != 2:
                            msg = f"Cannot convert to 2D array: {type(s)}"
                            raise ValueError(msg)  # noqa: TRY301
                        data.append(arr.copy() if copy else arr)
                    except Exception:
                        # If all conversions fail, raise the original error
                        msg = f"Cannot convert to 2D array: {type(s)}"
                        raise TypeError(msg) from e
            else:
                try:
                    arr = np.array(s)
                    if arr.ndim != 2:
                        msg = f"Cannot convert to 2D array: {type(s)}"
                        raise ValueError(msg)  # noqa: TRY301
                    data.append(arr.copy() if copy else arr)
                except Exception as e:
                    msg = f"Cannot convert to 2D array: {type(s)}"
                    raise TypeError(msg) from e

        return cls(np.array(data, dtype=object), dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def __getitem__(self, item):
        """Get images from an array."""
        if isinstance(item, (int, np.integer)):
            return self._data[item]  # Returns the np.ndarray or None
        else:
            # Slice, mask, or array of indices
            return ImageArray(self._data[item], dtype=self._dtype)

    def __len__(self):
        """Return the length of the array."""
        return len(self._data)

    @property
    def dtype(self):
        """Return the dtype of the array."""
        return self._dtype

    @property
    def nbytes(self):
        """Return the number of bytes consumed by the array."""
        # Sum of nbytes for each np.ndarray stored, plus the object array itself
        return self._data.nbytes + sum(
            arr.nbytes for arr in self._data if arr is not None
        )

    def isna(self):
        """Return a boolean array indicating missing values."""
        # Returns a boolean numpy array
        return np.array([x is None for x in self._data], dtype=bool)

    def take(self, indices, allow_fill=False, fill_value=None):
        """Take values from the array at the specified indices."""
        from pandas.core.algorithms import take as pandas_take

        if allow_fill and pd.isna(fill_value):  # Catches None, np.nan, pd.NA
            fill_value = self.dtype.na_value  # Use ImageDtype's na_value (None)

        result_data = pandas_take(
            self._data, indices, allow_fill=allow_fill, fill_value=fill_value
        )
        return ImageArray(result_data, dtype=self._dtype)

    def copy(self):
        """Return a copy of the array."""
        # Creates a new ImageArray with a copy of the underlying _data array.
        return ImageArray(self._data.copy(), dtype=self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # to_concat is a list of ImageArray instances
        concatenated_data = np.concatenate([arr._data for arr in to_concat])
        return cls(concatenated_data, dtype=to_concat[0].dtype)

    # Required method for pandas>=2.0
    def reshape(self, *args, **kwargs):
        """Reshape the array."""
        msg = "ImageArray does not support reshaping"
        raise NotImplementedError(msg)

    # Needed for pandas to work with our extension type
    def __array__(self, dtype=None):
        """Convert to a NumPy array."""
        return self._data

    def astype(self, dtype, copy=True):
        """Cast to a different dtype."""
        if isinstance(dtype, ImageDtype):
            if copy:
                return self.copy()
            return self
        return np.array(self._data, dtype=dtype)


@register_extension_dtype
class ImageDtype(ExtensionDtype):
    """Custom dtype to represent image data (2D numpy arrays)."""

    name = "image"  # Used for string representation, e.g., df.astype("image")
    type = np.ndarray  # The Python type for an individual element
    kind = "O"  # type: ignore # Object kind, as we're storing np.ndarray objects

    @classmethod
    def construct_from_string(cls, string):
        """Construct the dtype from a string."""
        if string == cls.name:
            return cls()
        else:
            msg = f"Cannot construct '{cls.__name__}' from '{string}'"
            raise TypeError(msg)

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return ImageArray

    @property
    def na_value(self):
        """The NA value for this dtype (None for object-stored np.ndarrays)."""
        return None


@pd.api.extensions.register_series_accessor("image")
class ImageAccessor:
    """Accessor for working with image data in Series.

    This accessor provides methods to easily work with image data stored in a pandas
    Series.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyref.image.extensions import ImageDtype
    >>>
    >>> # Convert a column to image dtype
    >>> df["RAW"] = df["RAW"].astype(ImageDtype())
    >>>
    >>> # Use the accessor to get the first image as a 2D numpy array
    >>> image = df["RAW"].image.stack()
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def array(self):
        """Get the underlying array."""
        if not isinstance(self._obj.dtype, ImageDtype):
            msg = "Can only use .array accessor with image dtype"
            raise TypeError(msg)
        return self._obj.array

    def stack(self, index=None):
        """Convert an image at the given index to a 2D numpy array.

        Parameters
        ----------
        index : int, optional
            Index of the image to stack. If None, stacks the first image.

        Returns
        -------
        np.ndarray
            The 2D numpy array representation of the image.
        """
        if not isinstance(self._obj.dtype, ImageDtype):
            # Try to convert to image dtype first
            try:
                img_series = self._obj.astype(ImageDtype())
                if index is None:
                    return np.stack(img_series.iloc[0])
                return np.stack(img_series.iloc[index])
            except Exception as e:
                msg = f"Cannot convert to image dtype: {e}"
                raise TypeError(msg) from e

        if index is None:
            return np.stack(self._obj.iloc[0])
        return np.stack(self._obj.iloc[index])

    def stack_all(self):
        """Convert all images in the series to a list of 2D numpy arrays.

        Returns
        -------
        list
            List of 2D numpy arrays.
        """
        if not isinstance(self._obj.dtype, ImageDtype):
            # Try to convert to image dtype first
            try:
                img_series = self._obj.astype(ImageDtype())
                return [np.stack(img) for img in img_series]
            except Exception as e:
                msg = f"Cannot convert to image dtype: {e}"
                raise TypeError(msg) from e

        return [np.stack(img) for img in self._obj]
