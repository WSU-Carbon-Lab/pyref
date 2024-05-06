"""Module for working with NEXAFS data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from periodictable import formula

# ---------------------------------------------------------------------
# Functions to read NEXAFS data from a file
# ---------------------------------------------------------------------
