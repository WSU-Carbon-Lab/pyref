"""
Expansion Module for xray reflectivity modeling and data reduction
in python.

@Author: Harlan Heilman
"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"
from pathlib import Path
from sys import modules

from matplotlib.pyplot import style

from .core import *
from .xrr import *

style_path = Path(__file__).parent / "themes.mplstyle"

style.use(style_path.as_posix())

del style_path
del modules["matplotlib.pyplot"]
del modules["pathlib"]
del style
del Path


# matplotlib.rcParams["mathtext.fontset"] = "stix"
# matplotlib.rcParams["font.family"] = "STIXGeneral"

# sns.set_theme(
#     rc={"figure.figsize": (10, 5), "axes.xmargin": 0.01, "axes.ymargin": 0.02}
# )
# plt.style.use(
#     "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
# )
