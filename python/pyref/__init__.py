"""
Expansion Module for xray reflectivity modeling and data reduction in python.

@Author: Harlan Heilman
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

from pyref.loader import PrsoxrLoader
from pyref.masking import InteractiveImageMasker
from pyref.pyref import py_read_experiment
from pyref.utils import err_prop_div, err_prop_mult, weighted_mean, weighted_std

__all__ = [
    "PrsoxrLoader",
    "err_prop_div",
    "err_prop_mult",
    "weighted_mean",
    "weighted_std",
    "ImageProcs",
    "InteractiveImageMasker",
    "py_read_experiment",
]
