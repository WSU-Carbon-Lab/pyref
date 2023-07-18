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
]

HEADER_DICT: Final[dict[str,str]] = {
    "Beamline Energy": "Energy",
    "Sample Theta": "Theta",
    "Beam Current": "Current",
    "Higher Order Suppressor": "HOS",
    "EPU Polarization": "POL",
}

REFL_COLUMN_NAMES: Final[dict] = HEADER_DICT | {
    "Beam Spot": "Intensity",
    "Dark Spot": "Background",
    "R": "Refl",
    "R Err": "Err",
    "Q": "Q",
}

POL: Final[dict] = {
    'P100': '100.0',
    'P190': '190.0',
}

FILE_NAMES = {
    "df": ".csv",
    "images": "_images.npz",
    "masked": "_masked.npz",
    "filtered": "_filtered.npz",
    "beamspot": "_beamspot.npz",
    "background": "_background.npz",
}