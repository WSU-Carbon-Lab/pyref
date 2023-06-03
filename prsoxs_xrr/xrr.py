"""Main module."""

import glob
import os
from typing import Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from numba import jit
from scipy.ndimage import median_filter
import uncertainties
from uncertainties import unumpy as unp
from xrr_toolkit import *
from collections import defaultdict


class XRR:
    """
    Main class for processing xrr data
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        mask: None | np.ndarray = None,
        height: int = 10,
        *args,
        **kwargs,
    ):
        self.directory = directory
        self.raw_data = RawData(directory, mask=mask, *args, **kwargs)
        self.images = Images(directory, mask=mask, height=height, *args, **kwargs)
        self.refl = Reflectivity(directory, *args, **kwargs)

        # properties
        self._mask = mask
        self._height = height

    def finalize(self):
        self.refl._finalize()

    def check_spot(
        self, spot_number: int, ylims: tuple | None = None, xlims: tuple | None = None
    ):
        self.raw_data.show_meta_info(spot_number + self.refl._izero_count)
        self.images.check_spot(spot_number + self.refl._izero_count)
        self.refl.highlight(spot_number, ylims=ylims, xlims=xlims)  # type: ignore

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask
        self.images = Images(self.directory, self._mask)
        self.refl = Reflectivity(self.directory)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height, *args, **kwargs):
        self._height = height
        self.images = Images(self.directory, mask=self.mask, height=self._height)
        self.refl = Reflectivity(self.directory)


#
#
#


class RawData:
    """
    Raw data object
    This object handles unpacking the fits files from each scan

    It contains one main function the _fits_loader
    This function uses the astropy background to load the metadata from the fits header into
    individual numpy arrays. Each numpy array is then stored as an attribute of the object

    """

    def __init__(self, directory: Union[str, os.PathLike], *args, **kwargs):
        # inheritance
        self.directory = directory

        # Constructed properties
        self.image_data: np.ndarray = []  # type: ignore
        self.header_data: np.ndarray = []  # type: ignore

        self.energies: np.ndarray = []  # type: ignore
        self.sample_theta: np.ndarray = []  # type: ignore
        self.beam_current: np.ndarray = []  # type: ignore
        self.q: np.ndarray = []  # type: ignore
        # object constructor
        self._fits_loader()

    def _fits_loader(self) -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory.

        Computing the xrr profile requires Beamline Energy, Sample Theta, and Beam Current
        Files are opened with astropy and those values are extracted from the header data
        The images are collected from the image data. The fits data is extracted into a numpy
        array
        """
        self.file_list = sorted(glob.glob(os.path.join(self.directory, "*.fits")))
        self.scan_name = self.file_list[0].split("\\")[-1].split("-")[0]

        arrays = [
            [
                fits.getheader(f, 0)["Beamline Energy"],
                fits.getheader(f, 0)["Sample Theta"],
                fits.getheader(f, 0)["Beam Current"],
                fits.getheader(f, 0)["Higher Order Suppressor"],
                fits.getheader(f, 0)["EPU Polarization"],
            ]
            for f in self.file_list
        ]

        self.image_data = np.squeeze(
            np.array([[fits.getdata(f, 2) for f in self.file_list]])
        )
        self.header_data = np.column_stack(arrays)

        (
            self.energies,
            self.sample_theta,
            self.beam_current,
            self.hos,
            self.polarization,
        ) = self.header_data
        self.q = scattering_vector(self.energies, self.sample_theta)

    def show_meta_info(self, scan_number):
        s = []

        s.append(f"Sample: {self.scan_name}")
        s.append(f"Scan Number: {scan_number}")
        s.append(f"Beam Energy: {self.energies[scan_number]:.2g}")
        s.append(f"Polarization: {int(self.polarization[scan_number] - 100)}")
        s.append(f"Beam Current: {self.beam_current[scan_number]:.2g}")
        s.append(f"Sample Theta: {self.sample_theta[scan_number]:.2g}")
        s.append(f"Higher Order Suppressor: {self.hos[scan_number]:.2g}")
        s.append("\n")
        print("\n".join(s))

    def save(self):
        file_name = self.directory.name  # type: ignore
        scan_info = {
            "Scan ID": self.scan_name,
            "Energy": self.directory.name,  # type: ignore
            "Number of frames": len(self.file_list),
        }
        import json

        with open(f"{self.directory / file_name}.json", "w", encoding="utf-8") as f:
            json.dump(scan_info, f, ensure_ascii=False, indent=4)
        total_data = np.column_stack(
            (self.sample_theta, self.energies, self.q, self.beam_current)
        )
        np.savetxt(
            f"{self.directory / file_name}",
            total_data,
            header="Sample Theta  Beamline Energy   Scattering Vector  Beam Current",
        )


#
#
#


class Images(RawData):
    """
    2D Image Object
    This object handles processing the 2D image data, this includes the following manipulations

    Locating the Beam: Finding the brightest pixel on a median filtered image

    ROI Reduction: Reduces the area of integration to a size x size rectangle

    Background Subtraction: Selects a representative dark rectangle of a similar size of the
    dark background

    Parameters
    ----------
    RawData : RawData
        Raw Data image for all fits files in the data directory

    Methods
    ----------
    generate_mask :
        Function returns a mask over the image using the same backend as the Roi reduction algorithm

    check_spot :
        Function that generates a plot depicting the images used to generate each data point in. This
        contains four plots:
        Raw image displaying the background rectangle, beam rectangle, and beam spot
        Median Filter image used for finding the beam spot
        Bright spot
        Dark spot
        Median Filtered bright spot
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        mask: None | np.ndarray = None,
        height: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(directory, mask, *args, **kwargs)
        # inhered properties
        self.images: np.ndarray = self.image_data
        self.filtered = median_filter(self.images, size=3)
        shape = (self.image_data.shape[0], height * 2 + 1, height * 2 + 1)  # type: ignore

        self.bright_spots: np.ndarray = np.empty(shape, dtype=np.int64)
        self.dark_spots: np.ndarray = np.empty(shape, dtype=np.int64)
        self.bright_sum: np.ndarray = np.empty(
            (self.image_data.shape[0]), dtype=np.int64
        )
        self.dark_sum: np.ndarray = np.empty((self.image_data.shape[0]), dtype=np.int64)

        if type(mask) is type(None):
            self.mask = None
            self.masked_image = self.images

        elif isinstance(mask, np.ndarray):
            self.mask = mask
            self.masked_image = np.squeeze(
                [np.multiply(image, mask) for image in self.filtered]  # type: ignore
            )

        else:
            raise TypeError("Mask must be a single numpy array")

        self._height = height
        self.reduced_roi: list = []
        self._roi_generator()

    def _roi_generator(self):
        """internal function to find the location of the beam spot on each frame"""
        for number, image in enumerate(self.images):
            masked_image = self.masked_image[number]

            max_idx = np.unravel_index(masked_image.argmax(), masked_image.shape)
            top: int = max_idx[0] - self._height
            bot: int = max_idx[0] + self._height + 1
            left: int = max_idx[1] - self._height
            right: int = max_idx[1] + self._height + 1

            if top < 0:
                top = 0
                bot = 2 * self._height + 1

            elif bot > image.shape[0]:
                bot = image.shape[0]
                top = image.shape[0] - (2 * self._height + 1)

            if left < 0:
                left = 0
                right = 2 * self._height + 1

            elif right > image.shape[1]:
                right = image.shape[1]
                left = image.shape[1] - (2 * self._height + 1)

            roi = [
                (
                    slice(top, bot),
                    slice(left, right),
                ),
                (
                    slice(image.shape[0] - bot, image.shape[0] - top),
                    slice(image.shape[0] - right, image.shape[0] - left),
                ),
            ]
            self.reduced_roi.append(roi)

            self.bright_spots[number] = image[roi[0]]
            self.dark_spots[number] = image[roi[1]]

        self.bright_sum = self.bright_spots.sum(axis=(1, 2))
        self.dark_sum = self.dark_spots.sum(axis=(1, 2))

    def _show_image_info(self, scan_number):
        """Build an info dump string that is printed"""

        signal_to_noise = self.bright_sum / self.dark_sum

        signal = (
            self.bright_sum[scan_number] - self.dark_sum[scan_number]
        ) / self.beam_current[scan_number]

        s = [
            f"Scattering Vector q: {self.q[scan_number]:.4g}",
            f"Bright Spot Intensity: {self.bright_sum[scan_number]:.4g}",
            f"Dark Spot Intensity: {self.dark_sum[scan_number]:.4g}",
            f"Absolute Signal: {signal:.4g}",
            f"Signal to Noise Ratio: {signal_to_noise[scan_number]:.4g}",
            "\n",
        ]
        print("\n".join(s))

    def generate_mask(self, scan_number):
        roi = self.reduced_roi[scan_number]
        self.mask = np.zeros(self.images[scan_number].shape)
        self.mask[roi[0]] = 1
        return self.mask

    def check_spot(self, scan_number):
        """external method for checking a scan, and a height"""
        self._show_image_info(scan_number)

        background_sub = np.maximum(
            (self.masked_image[scan_number] - self.dark_spots[scan_number].mean())
            / self.beam_current[scan_number],
            np.zeros(self.images[scan_number].shape),
        )

        style_kws = {
            "subplots": {"xticks": [], "yticks": []},
            "courser": {"colors": "blue", "lw": 0.8, "ls": "--"},
            "images": {"cmap": "hot", "norm": colors.LogNorm()},
            "horizontal": {"xmin": 0, "xmax": self.images[scan_number].shape[0] - 1},
            "vertical": {"ymin": 0, "ymax": self.images[scan_number].shape[1] - 1},
        }

        anchor_points = {
            "bright": tuple(
                (
                    self.reduced_roi[scan_number][0][0].start,
                    self.reduced_roi[scan_number][0][1].start,
                )
            ),
            "dark": tuple(
                (
                    self.reduced_roi[scan_number][1][0].start,
                    self.reduced_roi[scan_number][1][1].start,
                )
            ),
        }

        rect_bright = plt.Rectangle(  # type: ignore
            anchor_points["bright"],
            2 * self._height + 1,
            2 * self._height + 1,
            edgecolor="blue",
            facecolor="None",
        )

        rect_dark = plt.Rectangle(  # type: ignore
            anchor_points["dark"],
            2 * self._height + 1,
            2 * self._height + 1,
            edgecolor="black",
            facecolor="None",
        )

        """Build the return plot"""

        fig, ax = plt.subplots(1, 5, subplot_kw=style_kws["subplots"], figsize=(12, 12))
        axes = ["Raw", "Median", "Median", "Bright Spot", "Dark Spot"]
        for a, label in zip(ax, axes):
            a.set_xlabel(label)

        ax[0].imshow(self.images[scan_number], **style_kws["images"])

        ax[1].imshow(self.masked_image[scan_number], **style_kws["images"])

        ax[2].imshow(
            self.filtered[scan_number][self.reduced_roi[scan_number][0]],
            **style_kws["images"],
        )

        ax[3].imshow(self.bright_spots[scan_number], **style_kws["images"])

        ax[4].imshow(self.dark_spots[scan_number], **style_kws["images"])
        ax[1].add_patch(rect_dark)
        ax[1].add_patch(rect_bright)
        plt.show()


#
#
#


class Reflectivity(Images):
    """
    This object handles processing the image data into 1d traces that can be used for fitting and modeling in the future

    This object provides 4 main operations on the data

    Data Reduction: The 2d data is reduced to a single specular intensity and index of refraction q

    Normalization: Izero scans from the beginning of the data collection are used to normalize the data such that the
    direct beam measurement is unity

    Stitching: The data is collected in individual sample theta series. This allows the collection of data across
    6 orders of magnitude. The algorithm first finds these sample theta series packing them into individual numpy
    arrays. Then a stitching algorithm locates overlap points between these series using the specular intensity to
    calculate proper scale ratios to take into account the different intensities observed though each series.

    Parameters
    ----------
    Images : Images
        Image data object for each image in the directory

    Methods
    ----------
    plot : Plot
        Plots the computed reflectivity vs scattering vector q

    highlight: Plot
        Uses the plot backend but this function highlights a particular scan of interest

    plot_stitches: Plot
        Uses the plot backend to instead plot each q-series facilitating quick searching to determine what angles
        need more data.
    """

    def __init__(self, directory, *args, **kwargs):
        super().__init__(directory, *args, **kwargs)
        self.r = unp.uarray(
            (self.bright_sum - self.dark_sum) / self.beam_current,
            np.sqrt(self.bright_sum + self.dark_sum) / self.beam_current,
        )
        self._normalize()
        self._find_q_series()
        self._stitch_q_series()

    def _normalize(self):
        """Internal Function for normalizing reflectivity data"""

        self._izero_count = 0
        while True:
            if self.q[self._izero_count] == 0:
                self._izero_count += 1
            else:
                break

        izero = uaverage(
            self.r[: self._izero_count - 1] if self._izero_count > 0 else 1
        )
        izero_err = np.std(unp.std_devs(self.r[: self._izero_count - 1]))
        self.izero = ufloat(izero, izero_err)

        self.r = self.r[self._izero_count :] / self.izero
        self.q = self.q[self._izero_count :]

        assert self.r.size == self.q.size

    def _find_q_series(self, qtol: float = 1e-3):
        """Internal function for locating stitch points in the reflectivity data"""

        ind = np.where(np.diff(self.q) < -qtol)[0] + 1

        self.q_split = np.split(self.q, ind)

        self.r_split = np.split(self.r, ind)

    def _stitch_q_series(self, stol: float = 1e-1):
        """

        internal Function for stitching

        """
        self._ratios = np.empty(len(self.r_split))

        for i, (sub_q, sub_r) in enumerate(zip(self.q_split, self.r_split)):
            if i != len(self.q_split) - 1:
                ratio = np.array([])  # keep track of where the sections overlap

                for j, q in enumerate(sub_q):
                    stitch_indices = np.where(
                        np.isclose(self.q_split[i + 1], q, rtol=stol)
                    )
                    next_r_values = self.r_split[i + 1][stitch_indices]
                    r_value = sub_r[j]
                    ratio = np.append(ratio, r_value / next_r_values)
                ratio = uaverage(ratio)
                self.r_split[i + 1] = self.r_split[i + 1] * ratio
                self._ratios[i] = ratio
        self.r = np.concatenate(self.r_split)
        assert self.r.size == self.q.size

    def _finalize(self, qtol: float = 1e-1):
        """
        Internal function for finalizing the data
        """
        tally = defaultdict(list)
        for i, q in enumerate(self.q):
            val = round(q, 4)
            tally[val].append(i)

        index_list = sorted((key, locs) for key, locs in tally.items())
        self.q = np.array([val[0] for val in index_list])
        match_points = [val[1] for val in index_list]

        self.r_f = []
        for idx_set in match_points:
            rf = self.r[idx_set].mean()
            self.r_f.append(rf)
        self.r = np.array(self.r_f)
        assert self.r.size == self.q.size

    def plot(self, ylims: tuple | None = None, xlims: tuple | None = None):
        plt.yscale("log")
        plt.errorbar(
            self.q,
            unp.nominal_values(self.r),
            unp.std_devs(self.r),
            fmt=".",
        )
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel(r"q $[\AA^{-1}]$")
        plt.ylabel("Reflectivity")
        plt.show()

    def highlight(
        self,
        highlight,
        ylims: tuple | None = None,
        xlims: tuple | None = None,
        *args,
        **kwargs,
    ):
        plt.yscale("log")
        plt.errorbar(
            self.q,
            unp.nominal_values(self.r),
            unp.std_devs(self.r),
            fmt=".",
            *args,
            **kwargs,
        )
        plt.errorbar(
            self.q[highlight],
            unp.nominal_values(self.r[highlight]),
            unp.std_devs(self.r[highlight]),
            marker="s",
        )
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel(r"q $[\AA^{-1}]$")
        plt.ylabel("Reflectivity")
        plt.show()

    def plot_stitches(self, *args, **kwargs):
        plt.yscale("log")
        for i, (q, r) in enumerate(zip(self.q_split, self.r_split)):
            plt.errorbar(
                q,
                unp.nominal_values(r),
                unp.std_devs(r),
                fmt=".",
                label=f"q-series {i + 1}",
                *args,
                **kwargs,
            )
        plt.xlabel(r"q $[\AA^{-1}]$")
        plt.ylabel("Reflectivity")
        plt.legend()
        plt.show()


#
#
#

if __name__ == "__main__":
    dir = file_dialog()
    xrr1 = XRR(dir)
    xrr1.check_spot(1)
