"""Main module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from IPython.display import display
from ipywidgets import VBox, interactive

from pyref import pyref
from pyref.image import (
    apply_mask,
    beamspot,
    reduce_masked_data,
    reduction,
)
from pyref.masking import InteractiveImageMasker
from pyref.types import HeaderValue
from pyref.utils import err_prop_div, weighted_mean, weighted_std

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pyref.types import Motor
    from pyref.utils import IntoExprColumn

sns.set_theme(style="ticks")

intensity_agg = (pl.col("I [arb. un.]"), pl.col("δI [arb. un.]"))
overlap_agg = (
    pl.col("I[current]"),
    pl.col("δI[current]"),
    pl.col("I[prior]"),
    pl.col("δI[prior]"),
    pl.col("I₀ [arb. un.]"),
    pl.col("δI₀ [arb. un.]"),
)

current_columns = ["Sample Theta [deg]", "I [arb. un.]", "δI [arb. un.]"]
prior_columns = [
    "Sample Theta [deg]",
    "I [arb. un.]",
    "δI [arb. un.]",
    "I₀ [arb. un.]",
    "δI₀ [arb. un.]",
]
final_columns = [
    "θ [deg]",
    "I [arb. un.]",
    "δI [arb. un.]",
    "I₀ [arb. un.]",
    "δI₀ [arb. un.]",
    "Beamline Energy [eV]",
    "Q [Å⁻¹]",
    "r [a. u.]",
    "δr [a. u.]",
]


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
    files : list
        List of .fits to be loaded. Include full filepaths

        >>> #Recommended Usage
        >>> import pathlib
        >>> path_s = pathlib.Path('../ALS/2020 Nov/MF114A/spol/250eV')
        >>> files = list(path_s.glob('*fits')) # All .fits in path_s

        The newly created 'files' is now a list of filepaths to each reflectivity point.

    name : str
        Name associated with the dataset. Will be used when saving data.

    mask : np.ndarray (Boolean)
        Array with dimensions equal to an image. Elements set to `False` will be
        excluded when finding beamcenter.

    autoload : Boolean
        Set to false if you do not want to load the data upon creation of object.

    Attributes
    ----------
    name : str
        Human readable string that describes the dataset to be loaded. See 'name'
        parameter
    mask : np.ndarray (Bool)
        Data mask to be applied to all images.
    files : list
        List of filepaths
    shutter_offset : float
        Deadtime added to the piezo shutter exposure.
        >>> total_exposure = frame_exposure + shutter_offset
         It is recommended to measure at the time of taking data (see online tutorial)
    sample_location : int
        Sample location on the holder:
        Bottom of holder == 180, Top of holder == 0. Should be automatically updated
        when files load
    angle_offset : float
        Angle offset [deg] to be added to 'Sample Theta' to correctly calculate q.
        (see online tutorial)
    energy_offset : float
        Energy offset [eV] to be applied to 'Beamline Energy' to correctly calculate q.
    snr_cutoff : float
        snr is the ratio of light counts vs. dark counts for images used in calculated
        total signal.
        Any image found below this threshold will be excluded from processing.
        It is assumed that under this condition the beam is attenuated enough that we
        are unable to locate its center.
        Default value is 1.01 and is suitable for the ALS in most cases.
    variable_motors : list(str)
        List of upstream optics that were varied to modify flux during data collection.
        Defaults are Horizontal Exit Slit Size and Higher Order Suppressor
    imagex : int
        X-dimension of .fits. Will be automatically updated when files load.
    imagey : int
        Y-dimension of .fits. Will be automatically updated when files load
    edge_trim : tuple(int)
        Edge of the detector that you want to ignore in processing.
        Edge pixels can sometimes have higher background at longer exposures.
        Use this option to exclude them from finding the beamcenter.
        Typical value is (5, 5)
    darkside : 'LHS' or 'RHS'
        Side of the detector that you want to take dark image.
        The same size ROI will be chosen but offset to the edge of the detector.
        Will not include pixels cut by edge_trim.
    diz_threshold : int
        Dizinger intensity threshold to remove 'hot' pixels.
    diz_size : int
        Size of box to average to remove 'hot' pixels.

    Notes
    -----
    Print the loader to view variables that will be used in reduction. Update them
    using the attributes listed in this API.

    >>> loader = PrsoxrLoader(files, name='MF114A_spol')
    >>> print(loader) #Default values
        Sample Name - MF114A
        Number of scans - 402
        ______________________________
        Reduction Variables
        ______________________________
        Shutter offset = 0.00389278
        Sample Location = 0
        Angle Offset = -0.0
        Energy Offset = 0
        SNR Cutoff = 1.01
        ______________________________
        Image Processing
        ______________________________
        Image X axis = 200
        Image Y axis = 200
        Image Edge Trim = (5, 5)
        Dark Calc Location = LHS
        Dizinger Threshold = 10
        Dizinger Size = 3
    >>>loader.shutter_offset = 0.004 #Update the shutter offset
    >>>

    Once process attributes have been setup by the user, the function can be called to
    load the data. An ROI will need
    to be specified at the time of processing. Use the ``self.check_spot()`` function
    to find appropriate dimensions.

    >>> refl = loader(h=40, w=30)

    Data that has been loaded can be exported using the ``self.save_csv(path)`` and
    ``self.save_hdf5(path)`` functions.

    """

    def __init__(
        self,
        directory: Path,
        mask: np.ndarray | None = None,
    ):

        # Sample information
        self.name: str = directory.stem  # Name of the series to be loaded
        self.path: str = str(directory)  # Path to the data

        # Configuration
        self.shutter_offset: float = 0.00389278  # [s]
        self.angle_offset: float = 0  # [deg]
        self.energy_offset: float = 0  # [eV]
        self.snr_cutoff: float = (
            1.01  # SNR = I_refl/dark ~ Cutoff on if frame is ignored.
        )
        self.variable_motors: Motor = [
            HeaderValue.HIGHER_ORDER_SUPPRESSOR,
            HeaderValue.HORIZONTAL_EXIT_SLIT_SIZE,
        ]

        # Image stats
        self.edge_trim = 1  # Edge of detector to ignore

        # Files for output
        self.beam_drift = None
        self.meta: pl.DataFrame = pyref.py_read_experiment(
            str(directory), "xrr"
        ).with_columns(pl.col("Beamline Energy [eV]").round(1))
        self.data: pl.DataFrame | list[pl.DataFrame] = self.meta.group_by(
            "Beamline Energy [eV]"
        ).agg(pl.all())
        self.stitched: list[pl.DataFrame] = []
        self.refl: pl.DataFrame | None = None
        self.shape = len(self.meta)
        self.blur_strength: int = 4
        self.roi: int = 10
        self._energy: float | None = None
        self._polarization: Literal["s", "p"] | None = None

    @property
    def energy(self) -> list[float]:
        """Energy getter."""
        return np.sort(
            self.meta.select("Beamline Energy [eV]").unique().to_numpy().flatten()
        )

    @property
    def polarization(self) -> Literal["s", "p"]:
        """Polarization getter."""
        match self.meta.select("EPU Polarization [deg]").unique().to_numpy()[0][0]:
            case 190.0:
                return "p"
            case 100.0:
                return "s"
            case _:
                raise UnknownPolarizationError()

    def __str__(self):
        """Return string representation."""
        s = []  # ["{:_>50}".format("")]
        s.append(f"Sample Name - {self.name}")
        s.append(f"Number of scans - {len(self.files)}")
        s.append("{:_>30}".format(""))
        s.append("Reduction Variables")
        s.append("{:_>30}".format(""))
        s.append(f"Shutter offset = {self.shutter_offset}")
        s.append(f"Angle Offset = {self.angle_offset}")
        s.append(f"Energy Offset = {self.energy_offset}")
        s.append(f"SNR Cutoff = {self.snr_cutoff}")
        s.append("{:_>30}".format(""))
        return "\n".join(s)

    def __call__(
        self,
        roi: int | None = None,
        sample_detector_distance: float = 130,
        pixel_dim: float = 0.027,
    ):
        """Return reflectivity dataframe."""
        self.meta = pyref.py_simple_update(self.meta, self.path)
        refl = self._calc_refl(roi)
        print(refl)
        return refl

    def __len__(self):
        """Return the number of files."""
        return self.shape

    def check_spot(self):
        """Interactive plot for a selected frame with ROI and blur settings."""
        # Create sliders for frame index, ROI size, and blur strength
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
        energy_selector = widgets.Dropdown(
            options=self.energy,
            description="Energy",
            layout=widgets.Layout(width="200px"),
        )
        roi_selector = widgets.IntSlider(
            value=self.roi,
            min=0,
            max=50,
            step=1,
            description="ROI Size",
            layout=widgets.Layout(width="300px"),
        )
        blur_selector = widgets.IntSlider(
            value=self.blur_strength,
            min=0,
            max=50,
            step=1,
            description="Blur Strength",
            layout=widgets.Layout(width="300px"),
        )
        histogram = widgets.ToggleButton(
            value=False,
            description="Histogram",
            button_style="info",
            layout=widgets.Layout(width="300px"),
        )

        # Initialize figure for plotting
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(12.8, 4.8),
            gridspec_kw={"hspace": 0.5, "wspace": 0.5},
        )

        def plot_frame(blur, roi, energy, frame, histogram):
            """Inner function to process and plot a single frame with given settings."""
            # Get metadata and process the image for the specified frame
            meta = self.meta.filter(pl.col("Beamline Energy [eV]") == energy).row(
                frame, named=True
            )
            image = np.reshape(meta["Raw"], meta["Raw Shape"])[::-1]
            masked = apply_mask(image, self.mask)
            bs = beamspot(masked, roi, radius=blur)
            db, bg = reduction(
                image, beam_spot=bs, box_size=roi, edge_trim=self.edge_trim
            )

            imax = ax[0]
            hist = ax[1]
            # Clear and set up the plot
            imax.clear()
            hist.clear()

            cmap = plt.cm.get_cmap("terrain", 256)
            cmap.set_bad("black")

            imax.imshow(image, cmap=cmap, interpolation="none")

            # Draw arrow to beam spot labeling the properties
            bbox = {"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black", "lw": 1}
            arrowprops = {
                "arrowstyle": "->",
                "connectionstyle": "angle,angleA=0,angleB=90,rad=20",
            }
            props = f"direct beam = {db:.2f}\nbackground = {bg:.3f}\n"
            props += f"Q = {meta["Q [Å⁻¹]"]:.3f}\nE= {meta["Beamline Energy [eV]"]:.1f}"
            imax.annotate(
                props,
                xy=bs[::-1],
                xytext=(40, 10),
                bbox=bbox,
                arrowprops=arrowprops,
                textcoords="offset points",
            )

            # Add rectangles to highlight the beam spot and dark frame
            imax.add_patch(
                plt.Rectangle(
                    (0, bs[0] - roi),
                    image.shape[1],
                    2 * roi,
                    edgecolor="b",
                    facecolor="none",
                )
            )
            imax.add_patch(
                plt.Rectangle(
                    (bs[1] - roi, bs[0] - roi),
                    2 * roi,
                    2 * roi,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            imax.axis("off")
            # Add histogram of pixel intensities
            if histogram:
                # highlight the points that are at are within 100 of the max at
                # Slice the image to just the blue slice
                slice = image[bs[0] - roi : bs[0] + roi, :]
                df = pl.DataFrame(
                    {
                        "Intensity In ROI": slice.ravel(),
                    }
                )
                # Filter points that are within 100 of 2^16 (max pixel value) and
                # points within 100 of 0 (min pixel value)
                df = df.with_columns(
                    pl.when(pl.col("Intensity In ROI") > 2**16 - 10)
                    .then(pl.lit("High"))
                    .when(pl.col("Intensity In ROI") < 10)
                    .then(pl.lit("Low"))
                    .otherwise(pl.lit("Normal"))
                    .alias("Saturation")
                )
                sns.histplot(
                    data=df.to_pandas(),
                    x="Intensity In ROI",
                    stat="density",
                    ax=hist,
                    element="step",
                    hue="Saturation",
                )
                # add quanitiles
                q5 = np.quantile(slice, 0.01)
                q95 = np.quantile(slice, 0.99)
                q50 = np.quantile(slice, 0.5)
                hist.fill_betweenx(
                    hist.get_ylim(),
                    q5,
                    q95,
                    color="C5",
                    alpha=0.2,
                    label="99% of data",
                )

                hist.set_xlim(np.quantile(slice, 0.001), np.quantile(slice, 0.999))
                hist.axvline(q50, color="C5", linestyle="--")
                hist.set_title("Pixel Intensity Histogram")
                hist.legend()
            fig.canvas.draw_idle()

            # Update class attributes for blur and ROI
            self.blur_strength = blur
            self.roi = roi

            # Create an interactive widget to control the plot function

        interactive_plot = interactive(
            plot_frame,
            blur=blur_selector,
            roi=roi_selector,
            frame=frame_selector,
            energy=energy_selector,
            histogram=histogram,
        )

        # Display the sliders and plot
        display(VBox([interactive_plot]))
        plt.show()

    def mask_image(self):
        """Mask an image using the InteractiveImageMasker."""
        import matplotlib

        # set backend to use the widget backend
        matplotlib.use("module://ipympl.backend_nbagg")
        min_frame = self.meta.select(
            pl.col("Raw"),
            pl.col("Raw Shape"),
            pl.col("Raw")
            .map_elements(lambda img: img.sum(), return_dtype=pl.Float64)
            .alias("sum"),
        ).sort("sum")[0]
        image = np.reshape(
            min_frame["Raw"].to_numpy()[0], min_frame["Raw Shape"].to_numpy()[0][::-1]
        )[::-1, :]
        self.masker = InteractiveImageMasker(image)

    @property
    def mask(self):
        """Mask getter."""
        return self.masker.mask

    def write_csv(self, save_name):
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

        self.refl.write_csv(f"{self.path}/{save_name}_refl.csv")
        self.meta.write_csv(f"{self.path}/{save_name}_meta.csv")

    def write_hdf5(
        self, save_name: str, save_images=False, compress="gzip", en_offset=0
    ):
        """
        Export calculation to .hdf5 file.

        Parameters
        ----------
            path : str
                Directory that you want to save your data.

            hdf5_name : str
                Name of hdf5 file to save data.

            save_images : Boolean
                Option to save each image along with the processed data.

            compress : str
                Type of compression for image files.

            en_offset : float
                Optional offset to apply to naming convention. Use if energy offset was
                applied BEFORE taking data.

        Notes
        -----
        Able to save multiple scans to the same .hdf5 file by giving the same
        ``hdf5_name``. This allows you to compile all measurements on a single
        sample into a single file.
        The .hdf5 folder structure will be as follows::

            SAMPLE_NAME
                MEASUREMENT
                    EN_1 # Energy
                        POL_1 # Polarization
                            DATA
                                Q,R,R_err
                            META_DATA
                                IMAGE0 # Meta data and images
                                IMAGE1
                                ...
                        Pol_2
                            Data
                                ...
                    En_2
                        ...

        """
        if self.refl is None:
            print("Process data prior to saving it")
            return

        import h5py

        with h5py.File((f"{self.path}/{save_name}.h5py"), "a") as file_hdf5:
            # Create HDF5structure for saving data
            measurement = file_hdf5.require_group(
                "MEASUREMENT"
            )  # Folder to hold the data
            pol_label = "POL_" + str(int(self.meta["EPU Polarization"].iloc[0]))
            en_label = "EN_" + str(
                np.round(self.meta["Beamline Energy"].iloc[0] + en_offset, 1)
            ).replace(".", "pt")
            scan_label = measurement.require_group(en_label + "/" + pol_label)
            # Save Images if desired
            if save_images:
                raw_image_group = scan_label.require_group("META_DATA")
                for index, scan in enumerate(self.files):
                    image_folder = raw_image_group.require_group("IMAGE" + str(index))
                    image_folder.attrs["FILE_PATH"] = str(scan)
                    try:
                        image_folder.create_dataset(
                            "IMAGE_RAW", data=self.images[index], compression=compress
                        )
                        image_folder.create_dataset(
                            "IMAGE_LOCBEAM",
                            data=self.image_locbeam[index],
                            compression=compress,
                        )
                        image_folder.create_dataset(
                            "IMAGE_SPOT",
                            data=self.image_spot[index],
                            compression=compress,
                        )
                        image_folder.create_dataset(
                            "IMAGE_DARK",
                            data=self.image_dark[index],
                            compression=compress,
                        )
                    except RuntimeError:
                        pass
                    for key, value in self.meta.iloc[index].items():
                        image_folder.attrs[key] = value
            # Save process variables
            # data_group = scan_label.require_group('DATA')
            save_vars = self.process_vars
            for key in save_vars:
                scan_label.attrs[key] = str(save_vars[key])
            data_save = scan_label.create_dataset("DATA", data=self.refl)
            data_save.attrs["Energy label offset"] = str(en_offset)
            data_save.attrs["Column 1"] = "Q"
            data_save.attrs["Column 2"] = "R"
            data_save.attrs["Column 3"] = "R_err"

    def write_parquet(self, save_name: str):
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

        self.refl.write_parquet(f"{self.path}/{save_name}_refl.parquet")
        self.meta.write_parquet(f"{self.path}/{save_name}_meta.parquet")

    def _stitch(self, lzf: pl.LazyFrame):
        stitch_dfs = []
        # group by the HOS and HES columns
        for i, (_, stitch) in enumerate(
            lzf.collect().group_by(
                [
                    "Horizontal Exit Slit Size [um]",
                    "Higher Order Suppressor [mm]",
                    "EXPOSURE [s]",
                ],
                maintain_order=True,
            )
        ):
            if i == 0:
                izero = get_izero_df(stitch)
            else:
                prior = stitch_dfs[-1].filter(
                    pl.col("Sample Theta [deg]").is_in(stitch["Sample Theta [deg]"])
                )
                if prior.shape[0] == 0:
                    raise OverlapError(
                        selected=prior, prior=stitch_dfs[-1], current=stitch
                    )

                izero = get_reletive_izero(stitch, prior)
            stitch = stitch.join_asof(
                izero.select("Beamline Energy [eV]", "I₀ [arb. un.]", "δI₀ [arb. un.]"),
                on="Beamline Energy [eV]",
            )
            stitch_dfs.append(stitch)
        self.stitched.append(
            pl.concat(stitch_dfs).rename({"Sample Theta [deg]": "θ [deg]"})
        )

    def _calc_refl(
        self,
        roi: int | None = None,
    ):
        """
        Calculate Reflectivity from loaded data.
        """
        if roi is None:
            roi = self.roi
        lzf = reduce_masked_data(self.meta, roi, self.shutter_offset, self.snr_cutoff)
        for e in self.energy:
            self._stitch(lzf.filter(pl.col("Beamline Energy [eV]") == e))
        # Stitch the data together
        stitched = pl.concat(self.stitched)
        self.refl = stitched.with_columns(
            (pl.col("I [arb. un.]") / pl.col("I₀ [arb. un.]")).alias("r [a. u.]"),
            err_prop_div(
                *col_and_err("I [arb. un.]"),
                *col_and_err("I₀ [arb. un.]"),
            ).alias("δr [a. u.]"),
        ).select(final_columns)
        return self.refl

    def plot_data(self, energies: list[float] | float | None = None, line: bool = True):
        """Plot Reflectivity data."""
        if self.refl is None:
            print("Process data prior to plotting it")
            return
        if energies is None:
            energies = self.energy
        if isinstance(energies, (int, float)):
            energies = [energies]
        # ensure energies list is a subset of the available energies
        if not set(energies).issubset(self.energy):
            err = "Energies must be a subset of the available energies"
            raise ValueError(err)
        data = self.refl.filter(
            pl.col("Beamline Energy [eV]").is_in(energies),
            pl.col("Q [Å⁻¹]") > 0,
        ).to_pandas()
        sns.set_theme(style="ticks")
        if line:
            g = sns.lineplot(
                data=data,
                x="Q [Å⁻¹]",
                y="r [a. u.]",
                hue="Beamline Energy [eV]",
            )
        else:
            g = sns.scatterplot(
                data=data,
                x="Q [Å⁻¹]",
                y="r [a. u.]",
                hue="Beamline Energy [eV]",
            )
        # g.fill_between(self.refl["Q [Å⁻¹]"], lower, upper, alpha=0.3, color="C2")
        g.set_xlim(data["Q [Å⁻¹]"].min(), data["Q [Å⁻¹]"].max())
        g.set(yscale="log")
        return g


def _overlap_name_map(label: Literal["current", "prior"]) -> dict[str, str]:
    return {
        "I [arb. un.]": f"I[{label}]",
        "δI [arb. un.]": f"δI[{label}]",
    }


def col_and_err(col: str) -> tuple[IntoExprColumn, IntoExprColumn]:
    """Create a polars column exoression linking a column and its error."""
    return pl.col(col), pl.col(f"δ{col}")


def get_izero_df(stitch: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Direct Beam Intensity (Izero) for a given stitch.

    Parameters
    ----------
    stitch : pl.DataFrame
        The stitch to calculate the Izero for.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Izero and Izero Error.
    """
    i0 = stitch.filter(pl.col("Sample Theta [deg]") == 0.0)
    i0_val = i0.select("I [arb. un.]").mean().to_numpy()[0][0]
    i0_err = i0.select("I [arb. un.]").std().to_numpy()[0][0]
    return stitch.with_columns(
        pl.lit(i0_val).alias("I₀ [arb. un.]"),
        pl.lit(i0_err).alias("δI₀ [arb. un.]"),
    )


def rename_for_join(
    current_stitch: pl.DataFrame, prior_stitch: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Rename the columns of the DataFrames to be workable.

    Parameters
    ----------
    current_stitch : pl.DataFrame
        Current stitch dataframe that is being processed.
    prior_stitch : pl.DataFrame
        Previous stitch dataframe that is being processed.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        The DataFrames with the renamed columns.
    """
    current_stitch = current_stitch.select(current_columns).rename(
        _overlap_name_map("current")
    )
    prior_stitch = prior_stitch.select(prior_columns).rename(_overlap_name_map("prior"))
    return current_stitch, prior_stitch


def get_reletive_izero(
    current_stitch: pl.DataFrame, prior_stitch: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate the relative Izero for a given stitch.

    Parameters
    ----------
    current_stitch : pl.DataFrame
        Current dataframe being processed
    prior_stitch : pl.DataFrame
        Previous dataframe being processed

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the relative Izero and Izero Error.
    """
    # Ir = I0 * (I[current] / I[prior]).avg()
    energy = current_stitch["Beamline Energy [eV]"][0]
    current_stitch, prior_stitch = rename_for_join(current_stitch, prior_stitch)
    # Magic happens here do not touch thus unless you know what you are doing
    overlap = (
        (
            prior_stitch.tail(10).join(
                current_stitch,
                on="Sample Theta [deg]",
            )
        )
        .lazy()
        .group_by("Sample Theta [deg]", maintain_order=True)
        .agg(*overlap_agg)
        .with_columns(
            weighted_mean(pl.col("I[current]"), pl.col("δI[current]")).alias(
                "I[current]"
            ),
            weighted_mean(*col_and_err("I[prior]")).alias("I[prior]"),
            weighted_mean(*col_and_err("I₀ [arb. un.]")).alias("I₀ [arb. un.]"),
            weighted_std(*col_and_err("I[current]")).alias("δI[current]"),
            weighted_std(*col_and_err("I[prior]")).alias("δI[prior]"),
            weighted_std(*col_and_err("I₀ [arb. un.]")).alias("δI₀ [arb. un.]"),
        )
        .with_columns(
            (pl.col("I[current]") / pl.col("I[prior]")).alias("k"),
            err_prop_div(
                *col_and_err("I[current]"),
                *col_and_err("I[prior]"),
            ).alias("δk"),
        )
        .with_columns(
            (pl.col("I₀ [arb. un.]") * pl.col("k")).alias("I₀ʳ [arb. un.]"),
            err_prop_div(*col_and_err("I₀ [arb. un.]"), *col_and_err("k")).alias(
                "δI₀ʳ [arb. un.]"
            ),
            pl.lit(True).alias("dummy"),
        )
        .group_by("dummy")
        .agg(
            *col_and_err("I₀ʳ [arb. un.]"),
        )
        .with_columns(
            weighted_mean(*col_and_err("I₀ʳ [arb. un.]")).alias("I₀ [arb. un.]"),
            weighted_std(*col_and_err("I₀ʳ [arb. un.]")).alias("δI₀ [arb. un.]"),
            pl.lit(energy).alias("Beamline Energy [eV]"),
        )
    )
    return overlap.collect()
