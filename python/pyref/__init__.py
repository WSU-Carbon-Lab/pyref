"""
Xray Reflectivity data analysis package.

This package is designed to provide a simple interface for the analysis of X-ray
reflectivity data collected at the Advanced Light Source (ALS) at Lawrence Berkeley
National Laboratory (LBNL) Beamline 11.0.1.2. The package is based on the early
PRSoXR package written by Thomas Ferron during his time at NIST. This work makes a good
faith effort to provide the same functionality as the original PRSoXR package, but with
a more modern and user-friendly interface, and rust bindings for I/O operations.
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"

from pyref.io import read_experiment, read_fits
from pyref.loader import PrsoxrLoader
from pyref.masking import InteractiveImageMasker
from pyref.utils import err_prop_div, err_prop_mult, weighted_mean, weighted_std

__all__ = [
    "InteractiveImageMasker",
    "PrsoxrLoader",
    "err_prop_div",
    "err_prop_mult",
    "read_experiment",
    "read_fits",
    "weighted_mean",
    "weighted_std",
]
