"""
Expansion Module for xray reflectivity modeling and data reduction 
in python. 

@Author: Harlan Heilman
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

import matplotlib
import seaborn as sns

from ._config import *
from .core import *
from .image_manager import *
from .load_fits import *
from .refl_manager import *
from .refl_reuse import *
from .sorter import *
from .toolkit import *
from .xrr import *

sns.set_context("paper", font_scale=1.5)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

sns.set_theme(
    rc={"figure.figsize": (10, 5), "axes.xmargin": 0.01, "axes.ymargin": 0.02}
)
sns.set_palette("coolwarm")
