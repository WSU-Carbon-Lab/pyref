"""Functions to convert dataframes to refnx objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from refnx.dataset import ReflectDataset

if TYPE_CHECKING:
    import polars as pl


def to_reflect_dataset(
    df: pl.DataFrame, *, overwrite_err: bool = True
) -> ReflectDataset:
    """Convert a pandas dataframe to a ReflectDataset object."""
    if not overwrite_err:
        e = "overwrite_err=False is not implemented yet."
        raise NotImplementedError(e)
    Q = df["Q"].to_numpy()
    R = df["r"].to_numpy()
    dR = np.sqrt(np.square(R) + np.square(Q))

    return ReflectDataset((Q, R, dR))
