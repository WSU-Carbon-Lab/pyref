from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from astropy.io import fits

from pyref.core.config import AppConfig as config
from pyref.core.config import Value
from pyref.core.exceptions import FitsReadError

# --------------------------------------------------------
# Polars dataframe constructors for fits file data sources
# --------------------------------------------------------


def read_file(fits_file: str | Path, **kwargs) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a single fits file and returns a polars DataFrame."""
    with fits.open(fits_file) as hdul:
        _header = getattr(hdul[0], "header", None)
        image = getattr(hdul[2], "data", None)

    # ensure the fits files are not empty
    if _header is None:
        error = f"Could not read header from {fits_file}"
        raise FitsReadError(error)
    if image is None:
        error = f"Could not read image from {fits_file}"
        raise FitsReadError(error)

    # convert the header and image data to the correct types
    meta = {config.FITS_HEADER[key]: _header[key] for key in config.FITS_HEADER}
    meta["POL"] = "s" if meta["POL"] == 100 else "p"
    meta["ENERGY"] = round(meta["ENERGY"], 1)
    meta["HOS"] = round(meta["HOS"], 2)
    meta["DATE"] = pd.to_datetime(meta["DATE"])

    index = {
        "SAMPLE": fits_file.stem.split("-")[0],
        "SCAN_ID": int(fits_file.stem.split("-")[1].split(".")[0].lstrip("0")),
        "STITCH": 0,
    }
    meta |= index
    data = pl.DataFrame(index, schema_overrides=config.DATA_SHCHEMA)
    df_i = data.with_columns(
        SHAPE=pl.lit(image.shape), DATA=pl.lit(image.flatten().tolist())
    )

    df_m = pl.DataFrame(meta, schema_overrides=config.FITS_SCHEMA)
    return df_i, df_m


def read_scan(ccd_dir: str | Path, **kwargs) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Read a directory of fits files and returns a polars DataFrame."""
    fits_files = list(Path(ccd_dir).rglob("*.fits"))
    if not fits_files:
        error = f"No fits files found in {ccd_dir}"
        raise FitsReadError(error)

    df_m = []
    df_i = []
    stitch = 0
    for i, fits_file in enumerate(fits_files):
        data, meta = read_file(fits_file)
        # Check if the STICH needs to be updated
        if data.item(0, "THETA") == 0:
            stitch = 0

        elif (dfs[i - 1].item(0, "THETA") == 0) and (meta.item(0, "THETA") != 0):  # noqa
            stitch += 1
        elif (dfs[i - 1].item(0, "THETA") > meta.item(0, "THETA")) and (
            meta.item(0, "THETA") != 0
        ):
            stitch += 1
        data[0, "STITCH"] = stitch
        meta[0, "STITCH"] = stitch
        df_i.append(data)
        df_m.append(meta)
    df_i = pl.concat(df_i)
    df_m = pl.concat(df_m)
    return df_i.lazy(), df_m.lazy()


def scan_scan(ccd_dir: str | Path, **kwargs) -> pl.DataFrame:
    """Lazy implementation of read_scan including file watching."""
    err = "Not implemented yet."
    raise NotImplementedError(err)


# --------------------------------------------------------
# Classes for processing fits file data sources
# --------------------------------------------------------


@dataclass
class ExperimentLoader:
    """Class to load and process experiment data."""

    ccd_dir: str | Path

    shutter_offset = Value(0.0, "s")
    sample_location = Value(0, "deg")
    angle_offset = Value(0, "deg")
    energy_offset = Value(0, "eV")
    snr_cutoff = Value(1.01)  # Motors that were varied during data collection
    _process_height = Value(25, "pix")
    _process_width = Value(25, "pix")

    # Image stats
    edge_trim = Value(5, "pix")

    # Dezinger options
    diz_threshold = Value(10, "std")
    diz_size = Value(3, "pix")

    def __post_init__(self):
        self.ccd_dir = Path(self.ccd_dir)
        self.df_i, self.df_m = read_scan(self.ccd_dir)

    def __repr__(self) -> str:
        # String set to be
        term_size = 45
        s = f"{'─'*term_size}\n"
        s += "Experiment Loader\n"
        s += f"CCD Directory:      ┆{' '*4} {self.ccd_dir.stem}\n"
        s += f"{'═'*term_size}\n"
        s += f"Shutter Offset:     ┆{' '*4} {self.shutter_offset}\n"
        s += f"Sample Location:    ┆{' '*4} {self.sample_location}\n"
        s += f"Angle Offset:       ┆{' '*4} {self.angle_offset}\n"
        s += f"Energy Offset:      ┆{' '*4} {self.energy_offset}\n"
        s += f"SNR Cutoff:         ┆{' '*4} {self.snr_cutoff}\n"
        s += f"Process Height:     ┆{' '*4} {self._process_height}\n"
        s += f"Process Width:      ┆{' '*4} {self._process_width}\n"
        s += f"Edge Trim:          ┆{' '*4} {self.edge_trim}\n"
        s += f"Dezinger Threshold: ┆{' '*4} {self.diz_threshold}\n"
        s += f"Dezinger Size:      ┆{' '*4} {self.diz_size}\n"
        s += f"{'─'*term_size}\n"
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def anomaly_detection(self):
        """Detect anomalies in the experiment data."""
        self.check_stitch_dimension()

    def check_stitch_dimension(self):
        """
        Anomoly Detection for Stitches.

        Iterates though the data for the scan and ensures the dimension of the stitches
        are >=6. If the dimension is less than 2, the data is flagged in a new
        collumn called 'ANOMALY'.

        Returns
        -------
        None
        """
        grouped = self.df.group_by(["ENERGY", "STITCH"])
        # where the grouped.len()<4 flag it for anomaly
        flagged = grouped.agg(pl.len().alias("ANOMALY")).filter(pl.col("ANOMALY") < 2)
        self.df_m = self.df_m.join(flagged, on=["ENERGY", "STITCH"], how="left")

    def reload(self):
        """Reload the experiment data."""
        self.__post_init__()

    def preview(self, energy, n):
        """Preview the nth scan at a specific energy."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import mplcatppuccin
        import scienceplots

        mpl.style.use(["frappe", "science", "no-latex"])

        self.anomaly_detection()
        print(self)
        df = self.df.filter(pl.col("ENERGY") == energy).collect()
        shape = df.item(n, "SHAPE").to_numpy()
        image = df.item(n, "DATA").to_numpy()
        image = np.array(image).reshape(shape)
        plt.imshow(image)
        plt.show()

    def save(self):
        """Save the experiment data."""
        self.df.collect().write_parquet(self.ccd_dir / "data.parquet")

    def process(self):
        """Process the experiment data."""

    def extract_images(self):
        """Create a new image dataframe from the."""


if __name__ == "__main__":
    data_dir = config.DATA_DIR
    loader = ExperimentLoader(data_dir)
    loader.preview(283.7, 0)
    loader.save()
