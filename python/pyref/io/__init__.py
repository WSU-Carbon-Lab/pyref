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

Experiment catalog and discovery
-------------------------------
- :func:`~pyref.io.discover_fits`: Discover FITS files under a root path (flat or recursive).
- :func:`~pyref.io.parse_fits_stem`: Parse a FITS filename stem into sample_name, tag, experiment_number, frame_number.
- :func:`~pyref.io.build_catalog`: Build a per-file catalog DataFrame (names-only or with headers).
- :func:`~pyref.io.scan_view`: Aggregate catalog into a per-scan view (file_count, energy/Q range).
- :func:`~pyref.io.experiment_summary`: Quick view of an experiment directory (scan summary table).
- :func:`~pyref.io.filter_catalog_paths`: Filter catalog by sample_name, tag, or experiment number(s).

See the specific function documentation for more details on usage and parameters.
"""

from pyref.io.experiment_names import (
    ParsedFitsName,
    build_catalog,
    discover_fits,
    experiment_summary,
    filter_catalog_paths,
    parse_fits_stem,
    scan_view,
)
from pyref.io.readers import read_experiment, read_fits

__all__ = [
    "ParsedFitsName",
    "build_catalog",
    "discover_fits",
    "experiment_summary",
    "filter_catalog_paths",
    "parse_fits_stem",
    "read_experiment",
    "read_fits",
    "scan_view",
]
