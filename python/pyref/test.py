"""Testing."""

import polars as pl
import pyref_rs as rs

df = pl.DataFrame(
    rs.py_read_fits(
        "C:/Users/hduva/.projects/pyref/pyref/test/ZnPc_20nm_A85042-00001.fits"
    )
)

print(df)
print(df["Image"][0].len())
