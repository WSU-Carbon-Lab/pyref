"""
Xray Reflectivity data analysis package.

This package is designed to provide a simple interface for the analysis of X-ray
reflectivity data collected at the Advanced Light Source (ALS) at Lawrence Berkeley
National Laboratory (LBNL) Beamline 11.0.1.2. The package is based on the early
PRSoXR package written by Thomas Ferron during his time at NIST. This work makes a good
faith effort to provide the same functionality as the original PRSoXR package, but with
a more modern and user-friendly interface, and rust bindings for I/O operations.
"""

from pathlib import Path

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"

from pyref.io import fits_accessor  # noqa: F401 - registers df.fits accessor
from pyref.io import scan_experiment
from pyref.loader import PrsoxrLoader
from pyref.masking import InteractiveImageMasker
from pyref.utils import err_prop_div, err_prop_mult, weighted_mean, weighted_std


def get_data_path() -> Path:
    """Return the path to the package test/fixture data directory."""
    return Path(__file__).resolve().parent / "data"


__all__ = [
    "InteractiveImageMasker",
    "PrsoxrLoader",
    "err_prop_div",
    "err_prop_mult",
    "get_data_path",
    "read_experiment",
    "read_fits",
    "weighted_mean",
    "weighted_std",
]
