"""Tests for PrsoxrLoader catalog mode and on-demand reduction."""

from __future__ import annotations

import pytest

from pyref import get_data_path
from pyref.loader import PrsoxrLoader


def test_loader_metadata_only_and_reduce_at() -> None:
    data_dir = get_data_path()
    loader = PrsoxrLoader(data_dir, metadata_only=True)
    assert loader.meta is not None
    assert loader.meta.height >= 1
    assert "path" in loader.meta.columns
    refl = loader.reduce_at(0)
    assert refl.height == 1
    assert "r" in refl.columns
    assert "dr" in refl.columns
    assert "file_name" in refl.columns
