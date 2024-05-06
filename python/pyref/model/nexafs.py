"""Module for working with NEXAFS data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from periodictable import formula

# ---------------------------------------------------------------------
# Functions to read NEXAFS data from a file
# ---------------------------------------------------------------------


@dataclass
class Nexafs:
    """Class to store NEXAFS data."""

    file: Path | str
    data: pl.DataFrame | None = None
    angle: float = 55.4

    def __post_init__(self):
        if isinstance(self.file, str):
            self.file = Path(self.file)
        if not self.file.is_dir():
            self.data = pl.read_csv(self.file)
        else:
            msg = f"Cannot read data from directory: {self.file}"
            raise ValueError(msg)
