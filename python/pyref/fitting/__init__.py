"""Fitting module for pyref."""

from refnx._lib.emcee.moves.de import DEMove
from refnx._lib.emcee.moves.gaussian import GaussianMove
from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from pyref.fitting.logp import LogpExtra
from pyref.fitting.reflectivity import PXR_ReflectModel
from pyref.fitting.structure import PXR_SLD, PXR_MaterialSLD, PXR_NexafsSLD

demove = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7), 0.05)]
gmove = GaussianMove(1e-7)


__all__ = [
    "PXR_SLD",
    "CurveFitter",
    "DEMove",
    "GlobalObjective",
    "LogpExtra",
    "Objective",
    "PXR_MaterialSLD",
    "PXR_NexafsSLD",
    "PXR_ReflectModel",
    "ReflectDataset",
    "Transform",
    "demove",
    "gmove",
]
