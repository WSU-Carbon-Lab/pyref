"""Fitting module for pyref."""

from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from pyref.fitting.fitters import Fitter, LogpExtra, MultiEnergyFitter
from pyref.fitting.reflectivity import ReflectModel
from pyref.fitting.refnx_converters import to_reflect_dataset
from pyref.fitting.structure import SLD, MaterialSLD, NexafsSLD

__all__ = [
    "SLD",
    "CurveFitter",
    "Fitter",
    "GlobalObjective",
    "LogpExtra",
    "MaterialSLD",
    "MultiEnergyFitter",
    "NexafsSLD",
    "Objective",
    "ReflectDataset",
    "ReflectModel",
    "Transform",
    "to_reflect_dataset",
]
