from refnx._lib.emcee.moves.de import *
from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from pyref.fitting.logp import *

move = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7), 0.05)]
