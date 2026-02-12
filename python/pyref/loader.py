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

from pyref.image import apply_mask, locate_beam, reduction
from pyref.io.experiment_names import parse_fits_stem
from pyref.io.readers import get_image_corrected, read_experiment, read_fits
from pyref.masking import InteractiveImageMasker
from pyref.types import HeaderValue

DEFAULT_ROI = 10

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


def _mask_from_image(image: np.ndarray) -> np.ndarray:
    row_sums = np.sum(image, axis=1)
    total = np.sum(row_sums)
    if total == 0:
        return np.ones_like(image, dtype=bool)
    row_sums = row_sums / total
    cumsum = np.cumsum(row_sums[::-1])[::-1]
    lower_bound_index = int(np.argmax(cumsum <= 0.25))
    upper_bound_index = int(np.argmax(cumsum <= 0.95))
    mask = np.zeros_like(image, dtype=bool)
    mask[upper_bound_index:lower_bound_index, :] = True
    return mask


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
    directory : Path | None
        Path to the directory containing the experiment data. Required when
        paths is None. When paths is provided, used as experiment root for
        saving; if None, set to parent of first path.
    extra_keys : list[HeaderValue] | None, optional
        A list of extra header values to load from the data files, by default None.
    paths : list[Path] | list[str] | None, optional
        If provided, load only these FITS paths instead of discovering under directory.
        directory may be None in that case.
    """

    def __init__(
        self,
        directory: Path | None = None,
        extra_keys: list[HeaderValue] | None = None,
        paths: list[Path] | list[str] | None = None,
    ):
        default_keys: list[str] = [
            HeaderValue.BEAMLINE_ENERGY.hdu(),
            HeaderValue.EPU_POLARIZATION.hdu(),
            HeaderValue.SAMPLE_THETA.hdu(),
            "Sample Name",
            "Scan ID",
        ]
        if extra_keys:
            default_keys.extend([key.hdu() for key in extra_keys])

        if paths is not None:
            path_list = [Path(p).resolve() for p in paths]
            if not path_list:
                msg = "paths must not be empty."
                raise ValueError(msg)
            self.path = Path(directory).resolve() if directory is not None else path_list[0].parent
            meta = read_fits(path_list, headers=default_keys, engine="polars")  # type: ignore[arg-type]
            self.meta = cast("pl.DataFrame", meta)
        else:
            if directory is None:
                msg = "directory is required when paths is not provided."
                raise ValueError(msg)
            self.path = Path(directory).resolve()
            self.meta = cast(
                "pl.DataFrame",
                read_experiment(self.path, headers=default_keys, engine="polars"),
            )

        if "file_name" in self.meta.columns and "sample_name" not in self.meta.columns:
            stems = self.meta.get_column("file_name")
            sample_names: list[str] = []
            tags: list[str | None] = []
            exp_nums: list[int] = []
            frame_nums: list[int] = []
            for s in stems:
                parsed = parse_fits_stem(str(s))
                if parsed is None:
                    sample_names.append("")
                    tags.append(None)
                    exp_nums.append(0)
                    frame_nums.append(0)
                else:
                    sample_names.append(parsed.sample_name)
                    tags.append(parsed.tag)
                    exp_nums.append(parsed.experiment_number)
                    frame_nums.append(parsed.frame_number)
            self.meta = self.meta.with_columns(
                pl.Series("sample_name", sample_names),
                pl.Series("tag", tags),
                pl.Series("experiment_number", exp_nums),
                pl.Series("frame_number", frame_nums),
            )

        first_img = get_image_corrected(self.meta, 0)
        self.mask = _mask_from_image(np.asarray(first_img))
        refl_rows: list[dict[str, object]] = []
        for i in range(len(self.meta)):
            img = get_image_corrected(self.meta, i)
            img_arr = np.asarray(img)
            masked = apply_mask(img_arr, mask=self.mask)
            beam_spot = locate_beam(masked, DEFAULT_ROI)
            db_sum, bg_sum = reduction(masked, beam_spot, DEFAULT_ROI)
            r = float(db_sum / bg_sum) if bg_sum else 0.0
            dr = float(np.sqrt(r))
            row = self.meta.row(i, named=True)
            refl_rows.append({
                "file_name": row["file_name"],
                "EPU Polarization": row.get("EPU Polarization"),
                "Beamline Energy": row.get("Beamline Energy"),
                "Q": row.get("Q"),
                "r": r,
                "dr": dr,
            })
        self.refl = pl.DataFrame(refl_rows)
        self.shape = len(self.meta)
        self._polarization: Literal["s", "p"] | None = None
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
            img = get_image_corrected(self.meta, frame)
            image = np.asarray(img)
            masked = apply_mask(image, mask=self.mask)
            beam_spot = locate_beam(masked, DEFAULT_ROI)
            bs = (beam_spot[0], beam_spot[1])
            refl_row = self.refl.row(frame, named=True)
            meta = self.meta.row(frame, named=True)
            refl = refl_row["r"]

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
            q_val = meta.get("Q", refl_row.get("Q", 0.0)) or 0.0
            e_val = meta.get("Beamline Energy", refl_row.get("Beamline Energy", 0.0)) or 0.0
            props = (
                f"Reflectivity = {refl:.3e}\n"
                f"Q = {q_val:.3f}\n"
                f"E = {e_val:.1f}"
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
        frame_sums: list[float] = []
        for i in range(len(self.meta)):
            img = get_image_corrected(self.meta, i)
            frame_sums.append(float(np.sum(np.asarray(img))))
        min_idx = int(np.argmin(frame_sums))
        image = get_image_corrected(self.meta, min_idx)
        image = np.asarray(image)
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
