"""
I/O Operations for Reflectivity Data at BL 11.0.1.2.

This module provides a collection of tools for reading and writing data related to
X-ray reflectivity experiments, particularly those conducted at beamline 11.0.1.2.
The primary focus is on handling FITS files, which are commonly used for storing
astronomical and scientific image data.

The functions aim to provide a simple and consistent interface for loading experimental
data into standard data analysis structures like pandas or polars DataFrames.

Input/Output operations currently supported:
===========================================

FITS Files
----------
- :func:`~pyref.io.read_fits`: Read data from one or more FITS files.
- :func:`~pyref.io.read_experiment`: Read all FITS files from a directory or based on a
        pattern.

See the specific function documentation for more details on usage and parameters.
"""

from pyref.io.readers import read_experiment, read_fits

__all__ = ["read_experiment", "read_fits"]
