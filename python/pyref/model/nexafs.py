"""Module for working with NEXAFS data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import float64

import polars as pl
from periodictable import formula

# --------------------------------------------------------
# NEXAFS data classes
# --------------------------------------------------------


class TEYData:
    """Data class that describes total electron yield (TEY) data."""

    def to_asf(self) -> pl.DataFrame:
        """Converts TEY data to absorption strength factor (ASF)."""


class PEYData:
    """Data class that describes partial electron yield (PEY) data."""

    def to_asf(self) -> pl.DataFrame:
        """Converts PEY data to absorption strength factor (ASF)."""


class BLData:
    """Data class tht describes total beer lambert (BL) data."""

    def to_asf(self) -> pl.DataFrame:
        """Converts BL data to absorption strength factor (ASF)."""


class CSFile:
    """Class for working with csv files."""

    def read_file(self, file: str | Path) -> pl.DataFrame:
        """Reads a csv file and returns a polars DataFrame."""
        return pl.read_csv(file)

    def scan_file(self, file: str | Path) -> pl.DataFrame:
        """Scans a csv file and returns a polars DataFrame."""
        return pl.scan_csv(file)

    def write_file(self, file: str | Path, df: pl.DataFrame):
        """Writes a polars DataFrame to a csv file."""
        df.write_csv(file)


class TXTFile:
    """Class for working with txt files."""

    def read_file(self, file: str | Path) -> pl.DataFrame:
        """Reads a txt file and returns a polars DataFrame."""
        return pl.read_csv(file, separator="\t")

    def scan_file(self, file: str | Path) -> pl.DataFrame:
        """Scans a txt file and returns a polars DataFrame."""
        return pl.scan_txt(file, separator="\t")

    def write_file(self, file: str | Path, df: pl.DataFrame):
        """Writes a polars DataFrame to a txt file."""
        df.write_txt(file, separator="\t")


class PARQUETFile:
    """Class for working with parquet files."""

    def read_file(self, file: str | Path) -> pl.DataFrame:
        """Reads a parquet file and returns a polars DataFrame."""
        return pl.read_parquet(file)

    def scan_file(self, file: str | Path) -> pl.DataFrame:
        """Scans a parquet file and returns a polars DataFrame."""
        return pl.scan_parquet(file)

    def write_file(self, file: str | Path, df: pl.DataFrame):
        """Writes a polars DataFrame to a parquet file."""
        df.write_parquet(file)


@dataclass
class SingleAngle:
    """Data class for a single NEXAFS angle."""

    file: str | Path
    angle: float64
    data: pl.DataFrame

    def __post_init__(self):
        """Post initialization method."""
        self.angle = float64(self.angle)


@dataclass
class MultiAngle:
    """Data class for NEXAFS data."""

    file: str | Path
    angle: list[float64]
    data: pl.DataFrame


# --------------------------------------------------------
# Enums
# --------------------------------------------------------


class NEXAFSFileType(Enum):
    """Enumeration of supported NEXAFS file types."""

    CSV = "csv"
    TXT = "txt"
    PARQUET = "parquet"


class NEXAFSData(Enum):
    """Class for working with NEXAFS data."""

    SINGLE_ANGLE = SingleAngle
    MULTI_ANGLE = MultiAngle


class ExperimentType(Enum):
    """Enumeration of NEXAFS experiment types."""

    TEY = TEYData
    PEY = PEYData
    BL = BLData


# --------------------------------------------------------


@dataclass
class NEXAFS:
    """Data class for NEXAFS data."""

    data: list[NEXAFSData]
    formula: str
    edge: tuple[str, str]
    type: Literal["tey", "pey", "tfy"]
