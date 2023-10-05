from typing import Final, Literal


FLAGS: Final[dict] = {
    "-en": "Beamline Energy",
    "-pol": "EPU Polarization",
    "-n": "File Path",
}

HEADER_LIST: Final[list] = [
    "Beamline Energy",
    "Sample Theta",
    "Beam Current",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
]

HEADER_DICT: Final[dict[str, str]] = {
    "Beamline Energy": "Energy",
    "Sample Theta": "Theta",
    "Beam Current": "Current",
    "Higher Order Suppressor": "HOS",
    "EPU Polarization": "POL",
    "EXPOSURE": "Exposure",
}

REFL_COLUMN_NAMES: Final[dict] = HEADER_DICT | {
    "Images": "ims",
    "Masked": "masked",
    "Filtered": "filter",
    "Beam Image": "bs",
    "Dark Image": "ds",
    "Beam Spot": "int",
    "Dark Spot": "bg",
    "Raw": "adu",
    "R": "r",
    "R Err": "dr",
    "Q": "q",
    "i0": "i0",
    "i0Err": "di0",
    "Stat Update": "stat_update",
}

POL: Final[dict] = {
    "P100": "100.0",
    "P190": "190.0",
}

FILE_NAMES = {
    "meta.parquet": "_refl.csv",
    "image.parquet": "_image.parquet",
    ".json": "_refl.json.gzip",
}