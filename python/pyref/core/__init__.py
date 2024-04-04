from io import (
    ANGLES,
    HEADER_DICT,
    HEADER_LIST,
    NexafsIO,
    ReflIO,
)

from db import db
from frame import (
    AngleNexafs,
    OpticalConstant,
    OrientedOpticalConstants,
    ReflDataFrame,
)
from paths import FileDialog


def bt():
    """Return the path to the data directory."""
    from pathlib import Path

    path = (
        Path.home()
        / "Washington State University (email.wsu.edu)"
        / "Carbon Lab Research Group - Documents"
    )
    data = (
        path / "Synchrotron Logistics and Data" / "ALS - Berkeley" / "Data" / "BL1101"
    )
    return data
