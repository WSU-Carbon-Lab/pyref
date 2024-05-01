"""X-ray reflecitivity macro generation."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import HBox, VBox


@dataclass
class ExperimentalConfig:
    """Experimental configuration parameters."""

    # Experimental configuration parameters
    x_offset: ClassVar[float] = 0.15
    z_flip_pos: ClassVar[float] = 0.15
    reverse: ClassVar[bool] = False
    theta_offset: ClassVar[float] = 0
    point_density: ClassVar[int] = 12
    n_overlap: ClassVar[int] = 3
    n_uncert: ClassVar[int] = 4
    n_hos: ClassVar[int] = 1
    n_io: ClassVar[int] = 5

    # Sample Params
    z_direct: float
    effective_thick: float


@dataclass
class Stitch:
    """Stitch configuration parameters."""

    theta_i: float
    theta_f: float
    hos: float
    hes: float
    exposure: float


@dataclass
class EnergyConfig:
    """Configuration parameters needed for an energy."""

    energy: float
    pol: Literal["s", "p"]
    stiches: list[Stitch]


@dataclass
class SampleConfig:
    """Sample configuration parameters."""

    # Sample configuration parameters
    name: str
    energies: list[EnergyConfig]


def generate_macro():
    """Ipywidget enviroment to construct Configurations for the macro generator."""

    # Input Experimental Configuration Fields
    x_offset = widgets.FloatText(value=0.15, description="X Offset:", disabled=False)
    z_flip_pos = widgets.FloatText(
        value=0.15, description="Z Flip Pos:", disabled=False
    )
    reverse = widgets.Checkbox(value=False, description="Reverse:", disabled=False)
    theta_offset = widgets.FloatText(
        value=0, description="Theta Offset:", disabled=False
    )
    point_density = widgets.IntText(
        value=12, description="Point Density:", disabled=False
    )
    n_overlap = widgets.IntText(value=3, description="N Overlap:", disabled=False)
    n_uncert = widgets.IntText(value=4, description="N Uncert:", disabled=False)
    n_hos = widgets.IntText(value=1, description="N HOS:", disabled=False)
    n_io = widgets.IntText(value=5, description="N IO:", disabled=False)

    # Experimental Configuration Tab Boxes


def _generate_macro(
    sample: SampleConfig,
    experimental: ExperimentalConfig,
    energy: EnergyConfig,
    stitch: Stitch,
):
    """Generate the macro file."""
    # TODO: implement macro logic
