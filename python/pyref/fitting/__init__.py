"""Fitting module for pyref."""

from refnx.analysis import GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from pyref.fitting.fitters import (
    AnisotropyObjective,
    LogpExtra,
    ReflectModel,
    XrayReflectDataset,
    demove,
    gmove,
)
from pyref.fitting.fitters import (
    Fitter as CurveFitter,
)
from pyref.fitting.refnx_converters import to_reflect_dataset
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
    "gmove",
    "to_reflect_dataset",
]
