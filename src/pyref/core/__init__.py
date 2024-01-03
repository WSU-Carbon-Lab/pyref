import json

from .db import *
from .frame import *
from .io import *
from .paths import *

# init db

def db():
    with open(Path(__file__).parent / "config.json", "r") as f:
        path = json.load(f)["db"]
    return Path(path)

def bt():
    path = (
        Path.home()
        / "Washington State University (email.wsu.edu)"
        / "Carbon Lab Research Group - Documents"
    )
    data = path / "Synchrotron Logistics and Data" / "ALS - Berkeley" / "Data" / "BL1101"
    return data


