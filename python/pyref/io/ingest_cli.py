"""
Legacy entry point for ``pyref-ingest``; delegates to :mod:`pyref.cli`.
"""

from __future__ import annotations

from pyref.cli.main import main_ingest_shim as main

__all__ = ["main"]
