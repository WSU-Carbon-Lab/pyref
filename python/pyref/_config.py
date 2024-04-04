from typing import Final

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
    "DATE": "Date",
}

REFL_COLUMN_NAMES: Final[dict] = HEADER_DICT | {
    "Images": "Images",
    "Masked": "Masked",
    "Filtered": "Filtered",
    "Beam Image": "Beam",
    "Dark Image": "Dark",
    "Beam Spot": "Intensity",
    "Dark Spot": "Background",
    "Raw": "RawRefl",
    "R": "Refl",
    "R Err": "Err",
    "Q": "Q",
    "i0": "izero",
    "i0Err": "izeroErr",
}

POL: Final[dict] = {
    "P100": "100.0",
    "P190": "190.0",
}

FILE_NAMES = {
    "meta.parquet": "_refl.parquet.gzip",
    "image.parquet": "_image.parquet.gzip",
    ".json": "_refl.json.gzip",
}
