"""prsoxs_xrr package"""

__author__ = """Harlan Heilman"""
__email__ = "Harlan.Heilman@wsu.edu"
__version__ = "0.1.0"

from . _config import *
from . beam_spot_filter import *
from . data_sorter import *
from . image_manager import *
from . load_fits import *
from . refl_manager import *
from . refl_reuse import *
from . toolkit import *
from . xrr import *
from . display_path import *

import seaborn as sns
import matplotlib as mpl

sns.set_style('darkgrid')
sns.set_context('notebook')

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

import warnings
warnings.filterwarnings("ignore")