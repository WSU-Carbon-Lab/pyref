"""
Expansion Module for xray reflectivity modeling and data reduction
in python.

@Author: Harlan Heilman
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

# from . import toolkit, xrr
from .core import *
from .xrr import *

sns.set_context("paper", font_scale=1.5)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

sns.set_theme(
    rc={"figure.figsize": (10, 5), "axes.xmargin": 0.01, "axes.ymargin": 0.02}
)
plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)
