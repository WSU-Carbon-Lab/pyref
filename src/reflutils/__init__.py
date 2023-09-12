"""prsoxs_xrr package"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

from reflutils._config import *
from reflutils.beam_spot_filter import *
from reflutils.data_sorter import *
from reflutils.image_manager import *
from reflutils.load_fits import *
from reflutils.refl_manager import *
from reflutils.refl_reuse import *
from reflutils.toolkit import *
from reflutils.xrr import *
from reflutils.display_path import *

import seaborn as sns

sns.set_style(
    "white",
    rc = {
        "xtick.direction": "in", 
        "ytick.direction": "in", 
        "xtick.top":True, 
        "ytick.right":True, 
        "xtick.bottom" : True, 
        "ytick.left":True, 
        "grid.linestyle": "--"
        })
sns.set_context('notebook')