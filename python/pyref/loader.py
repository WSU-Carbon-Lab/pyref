"""Main module."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hvplot.polars  # noqa: F401
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display
from ipywidgets import VBox, interactive

from pyref.io.readers import read_experiment
from pyref.masking import InteractiveImageMasker
from pyref.types import HeaderValue

if TYPE_CHECKING:
    from typing import Literal


from typing import overload


@pl.api.register_series_namespace("image")
class ImageSeries:
    """
    A class to handle a series of images stored in a Polars Series.

    The series should contain arrays of type pl.Array(pl.Float64), where each array
    represents an image.
    """

    def __init__(self, series: pl.Series):
        self._s = series
        self._s = series

    def __getitem__(self, index: int):
        """
        Get the image at the specified index.

        Parameters
        ----------
        index : int
            The index of the image in the series.

        Returns
        -------
        np.ndarray: The image at the specified index.
        """
        return self._s[index].to_numpy()

    def counts(self, index: int) -> int:
        """
        Return the total counts of the image at the specified index.

        Parameters
        ----------
        index : int
            The index of the image in the series.

        Returns
        -------
        int: The total counts of the image.
        """
        img = self._s[index].to_numpy()
        return np.sum(img)

    @cached_property
    def argmin(self) -> int:
        """
        Return the index of the image with the minimal total counts.

        Return the index of the image with the minimal total counts.

        Returns
        -------
        int: The index of the image with the minimal total counts.
        """
        sums = self._s.map_elements(
            lambda img: np.sum(np.array(img)), return_dtype=pl.Int64
        )
        argmin = sums.arg_min()
        if argmin is None:
            msg = "The series is empty, cannot compute argmin."
            raise ValueError(msg)
        return argmin

    @cached_property
    def argmax(self) -> int:
        """
        Return the index of the image with the maximal total counts.

        Returns
        -------
        int: The index of the image with the maximal total counts.
        """
        sums = self._s.map_elements(
            lambda img: np.sum(np.array(img)), return_dtype=pl.Int64
        )
        argmax = sums.arg_max()
        if argmax is None:
            msg = "The series is empty, cannot compute argmax."
            raise ValueError(msg)
        return argmax

    @cached_property
    def min(self):
        """Return the image with the minimal total counts."""
        min_index = self.argmin
        return self[min_index]

    @cached_property
    def max(self):
        """Return the image with the maximal total counts."""
        max_index = self.argmax
        return self[max_index]

    def mean(self):
        """Return the mean image across all images in the series."""
        images = np.stack([np.array(img) for img in self._s])
        mean_image = np.mean(images, axis=0)  # type: ignore
        return mean_image.reshape((mean_image.shape[1], mean_image.shape[0]))

    @overload
    def cdf(self, image_stream: np.ndarray) -> np.ndarray: ...
    @overload
    def cdf(self, image_stream: int) -> np.ndarray: ...
    def cdf(self, image_stream) -> np.ndarray:
        """
        Something like the cumulative distribution function (CDF) of the image.
        """
        if isinstance(image_stream, int):
            # if an index is provided, get the image at that index
            selected_image = self[image_stream]
        elif isinstance(image_stream, np.ndarray):
            # if an image is provided, use it directly
            selected_image = image_stream
        else:
            msg = "image_stream must be an integer index or a numpy array."
            raise TypeError(msg)

        # calculate the row sums over the image
        row_sums = np.sum(selected_image, axis=1)
        row_sums /= np.sum(row_sums)  # normalize the row sums
        return row_sums

    def mask(self):
        """
        Create a mask for the image based on the CDF.
        """
        target = self.mean()
        cdf_values = self.cdf(target)
        cumsum = np.cumsum(cdf_values[::-1])[::-1]
        lower_bound_index = np.argmax(cumsum <= 0.25)
        upper_bound_index = np.argmax(cumsum <= 0.95)
        # create a mask to set to zero the values outside the bounds
        mask = np.zeros_like(target, dtype=bool)
        mask[upper_bound_index:lower_bound_index, :] = True
        return mask

    def imshow(self, index: int, **kwargs):
        """
        Display the given image using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            The image to display.
        **kwargs : dict
            Additional keyword arguments passed to plt.imshow().
        """
        plt.imshow(self[index], **kwargs)
        plt.title("Image")
        plt.show()


class UnknownPolarizationError(Exception):
    """Exception raised for unknown polarization values."""

    def __init__(self, message="Unknown polarization"):
        self.message = message
        super().__init__(self.message)


class OverlapError(Exception):
    """Exception raised when the overlap between two datasets is too small."""

    def __init__(
        self,
        message="No overlap found between datasets",
        selected=None,
        prior=None,
        current=None,
    ):
        self.message = message
        if prior is not None and current is not None:
            self.message += f"\nPrior: {prior.tail(10)}\nCurrent: {current.head(10)}"
            self.message += f"\nSelected: {selected}"
        super().__init__(self.message)


class PrsoxrLoader:
    """
    Class to load PRSoXR data from beamline 11.0.1.2 at the ALS.

    Parameters
    ----------
    directory : Path
        Path to the directory containing the experiment data.
    extra_keys : list[HeaderValue] | None, optional
        A list of extra header values to load from the data files, by default None.
    """

    def __init__(
        self,
        directory: Path,
        extra_keys: list[HeaderValue] | None = None,
    ):
        self.path: Path = Path(directory)
        default_keys: list[str] = [
            HeaderValue.BEAMLINE_ENERGY.hdu(),
            HeaderValue.EPU_POLARIZATION.hdu(),
            HeaderValue.SAMPLE_THETA.hdu(),
            "Sample Name",
            "Scan ID",
        ]
        if extra_keys:
            default_keys.extend([key.hdu() for key in extra_keys])

        self.meta: pl.DataFrame = cast(
            "pl.DataFrame",
            read_experiment(self.path, headers=default_keys, engine="polars"),
        )
        # using the mask
        self.refl: pl.DataFrame = self.meta.select(
            "file_name",
            "EPU Polarization",
            "Beamline Energy",
            "Q",
            pl.col("Simple Reflectivity").alias("r"),
            pl.col("Simple Reflectivity").sqrt().alias("dr"),
        )
        self.shape = len(self.meta)
        self._polarization: Literal["s", "p"] | None = None
        self.mask: np.ndarray = self.meta["RAW"].image.mask()  # type: ignore
        self.masker: InteractiveImageMasker | None = None

    @property
    def energy(self) -> list[float] | np.ndarray:
        """Energy getter."""
        return np.sort(
            self.meta.select("Beamline Energy").unique().to_numpy().flatten()
        )

    @property
    def polarization(self) -> Literal["s", "p"]:
        """Polarization getter."""
        pol_val = self.meta.select("EPU Polarization").unique().to_numpy()[0][0]
        if pol_val == 190.0:
            return "p"
        elif pol_val == 100.0:
            return "s"
        else:
            raise UnknownPolarizationError()

    @property
    def name(self) -> str | list[str] | np.ndarray:
        """Name getter."""
        return (
            self.meta.filter(~pl.col("Sample Name").str.starts_with("Captured"))
            .select("Sample Name")
            .unique()
            .to_numpy()
            .flatten()
        )

    @property
    def scan_id(self) -> int | list[int] | np.ndarray:
        """Scan ID getter."""
        return (
            self.meta.filter(~pl.col("Sample Name").str.starts_with("Captured"))
            .select("Scan ID")
            .unique()
            .to_numpy()
            .flatten()
        )

    @property
    def files(self) -> str | list[str] | np.ndarray:
        """Files getter."""
        return self.meta.select("file_name").unique().to_numpy().flatten()

    def __str__(self):
        """Return string representation."""
        s = []  # ["{:_>50}".format("")]
        s.append(f"Sample Name - {self.name}")
        s.append(f"Number of scans - {len(self.files)}")
        return "\n".join(s)

    def __call__(
        self,
    ):
        """Return reflectivity dataframe."""
        return self.refl

    def __len__(self):
        """Return the number of files."""
        return self.shape

    def check_spot(self):
        """Interactive plot for a selected frame with ROI and blur settings."""
        import matplotlib

        matplotlib.use("module://ipympl.backend_nbagg")
        frame_selector = widgets.IntSlider(
            min=0,
            max=self.shape - 1,
            step=1,
            description="Frame Index",
            layout=widgets.Layout(width="300px"),
            continuous_update=False,
        )

        # Initialize figure for plotting
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6.4, 4.8),
        )

        def plot_frame(frame):
            """Inner function to process and plot a single frame with given settings."""
            meta = self.meta.row(frame, named=True)

            image = self.meta["RAW"].image[frame]  # type: ignore
            bs = (meta["Simple Spot Y"], meta["Simple Spot X"])
            # cast the beamspot into a corrected tuple indicting the (x,y) coordinates
            # after the image is reshaped
            refl = meta["Simple Reflectivity"]

            # Clear and set up the plot
            ax.clear()

            cmap = plt.cm.get_cmap("terrain", 256)
            cmap.set_bad("black")

            ax.imshow(image, cmap=cmap, interpolation="none")

            # Draw arrow to beam spot labeling the properties
            bbox = {"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black", "lw": 1}
            arrowprops = {
                "arrowstyle": "->",
                "connectionstyle": "angle,angleA=0,angleB=90,rad=20",
            }
            props = (
                f"Reflectivity = {refl:.3e}\n"
                f"Q = {meta['Q']:.3f}\n"
                f"E = {meta['Beamline Energy']:.1f}"
            )
            ax.annotate(
                props,
                xy=bs[::-1],
                xytext=(40, 10),
                bbox=bbox,
                arrowprops=arrowprops,
                textcoords="offset points",
            )
            ax.axis("off")
            fig.canvas.draw_idle()

        interactive_plot = interactive(
            plot_frame,
            frame=frame_selector,
        )

        # Display the sliders and plot
        display(VBox([interactive_plot]))
        plt.show()

    def mask_image(self):
        """
        Interactively mask an image using a rectangular selector.

        This method opens a window with an image and allows the user to draw a
        rectangle to create a mask. The initial mask is based on the automatically
        generated mask.
        """
        # set backend to use the widget backend
        matplotlib.use("module://ipympl.backend_nbagg")
        min_frame = self.meta.select(
            pl.col("SUBTRACTED").is_not_null(),
            pl.col("SUBTRACTED")
            .map_elements(lambda img: np.sum(img), return_dtype=pl.Int64)
            .alias("sum"),
        ).sort("sum")[0]
        image = min_frame["SUBTRACTED"].to_numpy()
        self.masker = InteractiveImageMasker(image, mask=self.mask)
        self.mask = self.masker.get_mask()

    def write_csv(self):
        """
        Save calculated reflectivity as a .csv file.

        Parameters
        ----------
            path : str
                Directory that you want to save your data.

            save_name : str
                Name of output file

            save_meta : Boolean
                Option to save compilation of meta data along with reflectivity


        Notes
        -----
        Will create a folder /meta_data/ if it does not exist in 'path' directory
        to save meta_data
        """
        if self.refl is None:
            print("Process data prior to saving it")
            return

        for g in self.refl.group_by("file_name"):
            name = g[0][0]
            data = g[1].drop("file_name")
            data.drop_in_place("file_name")
            data.write_csv(f"{self.path}/{name}.csv")

    def write_parquet(self):
        """
        Save calculated reflectivity as a .parquet file.

        Parameters
        ----------
            path : str
                Directory that you want to save your data.

            save_name : str
                Name of output file

        Notes
        -----
        Will create a folder /meta_data/ if it does not exist in 'path' directory
        to save meta_data
        """
        if self.refl is None:
            print("Process data prior to saving it")
            return
        refl = self.path / "refl"
        meta = self.path / "meta"

        if not refl.exists():
            refl.mkdir()

        if not meta.exists():
            meta.mkdir()

        self.refl.write_parquet(refl / "all.parquet")
        self.meta.write_parquet(meta / "all.parquet")

        for g in self.refl.group_by("file_name"):
            name = g[0][0]
            data = g[1].drop("file_name")
            data.write_parquet(refl / f"{name}.parquet")
            g_meta = self.meta.filter(pl.col("file_name") == name)
            g_meta.write_parquet(meta / f"{name}.parquet")

    def plot_data(self):
        """Plot Reflectivity data."""
        if self.refl is None:
            print("Process data prior to plotting it")
            return

        refl = self.refl.filter(pl.col("Q").gt(0.0) & pl.col("r").gt(0.0))

        p = refl.hvplot.scatter(  # type: ignore
            x="Q",
            y="r",
            by=["file_name", "Beamline Energy"],
            title="Reflectivity",
            height=600,
            width=1200,
            muted_alpha=0,
        ).opts(
            legend_position="top_right",
            logy=True,
            xlim=(0, refl["Q"].max() * 1.1),  # type: ignore
        )
        return p
