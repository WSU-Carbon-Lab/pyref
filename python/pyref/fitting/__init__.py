"""Fitting module for pyref."""

from refnx.analysis import GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from pyref.fitting.fitters import (
    AnisotropyObjective,
    LogpExtra,
    ReflectModel,
    demove,
)
from pyref.fitting.fitters import (
    Fitter as CurveFitter,
)
from pyref.fitting.io import XrayReflectDataset
from pyref.fitting.structure import SLD, MaterialSLD, UniTensorSLD

__all__ = [
    "SLD",
    "AnisotropyObjective",
    "CurveFitter",
    "GlobalObjective",
    "LogpExtra",
    "MaterialSLD",
    "Objective",
    "ReflectDataset",
    "ReflectModel",
    "Transform",
    "UniTensorSLD",
    "XrayReflectDataset",
    "demove",
]
