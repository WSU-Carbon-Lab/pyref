"""
NEXAFS (Near-Edge X-Ray Absorption Fine Structure) IO and analysis.

Provides directory-based loading of NEXAFS scan files, multiple normalization
schemes (bare atom, constant background, Si/O background, polynomial), and an
interactive Jupyter widget for pre/post edge selection and normalization preview.
"""

from pyref.nexafs.directory import (
    NexafsDirectory,
    quality_display_symbol,
    quality_level_from_rms,
)
from pyref.nexafs.io import load_nexafs
from pyref.nexafs.normalization import NormalizationScheme
from pyref.nexafs.widget import NexafsWidget

__all__ = [
    "NexafsDirectory",
    "NexafsWidget",
    "NormalizationScheme",
    "load_nexafs",
    "quality_display_symbol",
    "quality_level_from_rms",
]
