"""prsoxs_xrr package"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

from ._config import *
from .image_manager import *
from .load_fits import *
from .refl_manager import *
from .refl_reuse import *
from .toolkit import *
from .xrr import *

import seaborn as sns

sns.set_style(
    "white",
    rc={
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "grid.linestyle": "--",
    },
)
sns.set_context("notebook")
