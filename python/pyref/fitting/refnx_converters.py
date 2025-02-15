"""Functions to convert dataframes to refnx objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from refnx.dataset import ReflectDataset

if TYPE_CHECKING:
    import polars as pl


def to_reflect_dataset(
    df: pl.DataFrame, *, gb_energy="Beamline Energy [eV]", overwrite_err: bool = True
) -> ReflectDataset | list[ReflectDataset]:
    """Convert a pandas dataframe to a ReflectDataset object."""
    if not overwrite_err:
        e = "overwrite_err=False is not implemented yet."
        raise NotImplementedError(e)
    datasets = []
    for _, g in df.group_by(gb_energy):
        Q = g["Q"].to_numpy()
        R = g["r"].to_numpy()
        # Calculate initial dR
        dR = 0.15 * R + 0.5e-6 * Q
        # Ensure dR doesn't exceed 90% of R to keep R-dR positive
        dR = np.minimum(dR, 0.9 * R)
        ds = ReflectDataset(data=(Q, R, dR))
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return datasets


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import polars as pl

    df = pl.read_parquet("/home/hduva/projects/data/june_processed.parquet")
    ds = to_reflect_dataset(df.filter(pl.col("sample").str.starts_with("mono")))

    for s in ds:
        s.plot()
        plt.yscale("log")
        plt.show()
