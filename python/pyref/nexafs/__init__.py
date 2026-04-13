"""
NEXAFS normalization, DataFrame accessor (df.nexafs), and database load.
"""

from pyref.nexafs.accessor import (
    NexafsAccessor,
    normalize_by_group,
)
from pyref.nexafs.io import load_nexafs

__all__ = ["NexafsAccessor", "load_nexafs", "normalize_by_group"]
