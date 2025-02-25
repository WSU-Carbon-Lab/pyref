"""Functions to convert dataframes to refnx objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyref.fitting.reflectivity import XrayReflectDataset

if TYPE_CHECKING:
    import polars as pl


def to_reflect_dataset(
    df: pl.DataFrame,
    *,
    r_percent=0.05,
    q_percent=0.5e-6,
    gb_energy="Beamline Energy [eV]",
    overwrite_err: bool = True,
) -> XrayReflectDataset:
    """Convert a pandas dataframe to a ReflectDataset object."""
    Q = df["Q"].to_numpy()
    R = df["r"].to_numpy()
    # Calculate initial dR
    dR = r_percent * R + q_percent * Q
    # Ensure dR doesn't exceed 90% of R to keep R-dR positive
    dR = np.minimum(dR, 0.9 * R)
    ds = XrayReflectDataset(data=(Q, R, dR))
    return ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import polars as pl

    df = pl.read_parquet("/home/hduva/projects/data/june_processed.parquet")
    ds = to_reflect_dataset(df.filter(pl.col("sample").str.starts_with("mono")))

    for s in ds:
        s.plot()
        plt.yscale("log")
        plt.show()
