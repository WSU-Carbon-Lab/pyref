"""Main module."""

from numbers import Rational
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy
import glob
from matplotlib.colors import LogNorm

from xrr_toolkit import *


class XRR:
    """
    Main class for processing xrr data
    """

    def __init__(self, directory: str, *args, **kwargs):
        # main properties
        self.directory = directory

        self.r = np.array([], dtype=np.float64)
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None

        # hidden properties
        self._error_method = error_method

    def calc_xrr(self, *args, **kwargs):
        """Main process for loading and calculating reflectivity curves"""
        self._fits_loader()
        self._data_reduction(*args, **kwargs)
        self._normalize()
        self._stitch(*args, **kwargs)
        self.r = np.concatenate(self.r_split, axis=None)

    def _fits_loader(self) -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory.

        Computing the xrr profile requires Beamline Energy, Sample Theta, and Beam Current
        Files are opened with astropy and those values are extracted from the header data
        The images are collected from the image data

        """
        file_list = sorted(glob.glob(os.path.join(self.directory, "*.fits")))

        arrays = [
            [
                fits.getheader(f, 0)["Beamline Energy"],
                fits.getheader(f, 0)["Sample Theta"],
                fits.getheader(f, 0)["Beam Current"],
            ]
            for f in file_list
        ]

        self.images = np.squeeze(np.array([[fits.getdata(f, 2) for f in file_list]]))
        (
            self.energies,
            self.sample_theta,
            self.beam_current,
        ) = np.column_stack(arrays)

        self.q = scattering_vector(self.energies, self.sample_theta)

    def _image_slicer(self):
        max_idx = np.unravel_index(image.argmax(), image.shape)
        _min = np.array(max_idx) - HEIGHT
        _max = np.array(max_idx) + HEIGHT

        correction = np.zeros(2, dtype=np.int64)
        if np.any(_min < 0):
            loc = np.where(_min < 0)
            correction[loc] = -1 * _min[loc]
        if np.any(_max > image.shape[0]):
            loc = np.where(_max > image.shape[0])
            correction[loc] = image.shape[0] - _max[loc] - 1
        else:
            pass
        new_idx = tuple(map(lambda x, y: np.int64(x + y), max_idx, correction))
        assert is_valid_index(image, new_idx)

        roi = (
            slice(new_idx[0] - HEIGHT, new_idx[0] + HEIGHT + 1),
            slice(new_idx[1] - HEIGHT, new_idx[1] + HEIGHT + 1),
        )

    def _data_reduction(self) -> None:
        bright_spots, dark_spots = zip(*[locate_spot(image) for image in self.images])
        bright_spots = np.stack(bright_spots)
        dark_spots = np.stack(dark_spots)

        bright_sum = bright_spots.sum(axis=(2, 1))
        dark_sum = dark_spots.sum(axis=(2, 1))

        dark_std = np.array([np.std(np.ravel(u)) for u in dark_spots])

        r = (bright_sum - dark_sum) / self.beam_current
        self.std_err = dark_std / np.sqrt(
            bright_spots.shape[0]
        )  # only includes std of dark not of bright spot
        self.shot_err = np.sqrt(bright_sum + dark_sum) / self.beam_current
        if self._error_method == "shot":
            self.r = unumpy.uarray(r, self.shot_err)
        elif self._error_method == "std":
            self.r = unumpy.uarray(r, self.std_err)
        else:
            raise Exception(
                'Choose "shot" for possonian statistics, or "std" for gaussian statistics'
            )
        assert self.q.size == self.r.size

    def _normalize(self):
        """
        Normalization
        """
        # find the cutoff for i_zero points
        izero_count = np.count_nonzero(self.q == 0)

        izero = uaverage(self.r[: izero_count - 1]) if izero_count > 0 else ufloat(1, 0)

        self.r = self.r[izero_count:] / izero
        self.q = self.q[izero_count:]
        self.images = self.images[izero_count:]
        assert self.r.size == self.q.size

    def _update_stats(self):
        """
        Internal function to update the stats based on the first frame in the data set.
        """
        self._repeat_index = np.where(np.absolute(np.diff(self.q)) < 1e-3)[0] + 1

    def _calculate_stitch_points(self):
        """
        Internal function for computing the stitch points in the dataset.
        The function splits self.r, self.q, and self.images into
        self.r_split, self.q_split, and self.image_split
        """

    def _stitch(self, tol=0.1):
        """
        Internal function used to stitch each of the subsets together.
        The function uses the backend intersect1d numpy function to
        determine intersct points between subsequent sets

        """
        ind = np.where(np.diff(self.q) < 0)[0] + 1

        q_split = np.split(self.q, ind)
        r_split = np.split(self.r, ind)
        self._image_split = np.split(self.images, ind)

        stitch_points = min([len(sub) for sub in q_split]) + 1

        self.q_split = [sub for sub in q_split if len(sub) > stitch_points]
        self.r_split = [sub for sub in r_split if len(sub) > stitch_points]
        self._ratios = []
        self._overlap = []
        for i, (sub_q, sub_r) in enumerate(zip(self.q_split, self.r_split)):
            if i == len(self.q_split) - 1:
                pass
            else:
                ratio = np.array([])  # keep track of where the sections overlap
                for j, q in enumerate(sub_q):
                    stitch_indices = np.where(
                        np.isclose(self.q_split[i + 1], q, rtol=tol)
                    )[0]
                    next_r_values = self.r_split[i + 1][stitch_indices]
                    r_value = sub_r[j]
                    ratio = np.append(ratio, r_value / next_r_values)
                ratio = uaverage(ratio)
                self.r_split[i + 1] = self.r_split[i + 1] * ratio
                self._ratios.append(ratio)
        self.r = np.concatenate(self.r_split)
        self.q = np.concatenate(self.q_split)
        assert self.r.size == self.q.size

    def plot(self) -> None:
        """Plot the X-ray reflectometry data and a reference curve."""
        plt.errorbar(
            self.q, unumpy.nominal_values(self.r), unumpy.std_devs(self.r), fmt="."
        )
        plt.legend()
        plt.yscale("log")
        plt.show()

    def full_plot(self):
        import seaborn as sns

        fig, axs = plt.subplots(3, 1, sharex=True)
        plt.tick_params(which="both", right=True, top=True)
        plt.minorticks_on()

        for j, method in enumerate(self.METHODS):
            self.error_method = method
            self.calc_xrr()
            for i, sub in enumerate(zip(self.q_split, self.r_split)):
                axs[1 + j].errorbar(
                    sub[0],
                    unumpy.nominal_values(sub[1]),
                    unumpy.std_devs(sub[1]),
                    fmt=".",
                    label=f"Scan = {i+1}",
                )
                axs[1 + j].set_ylabel(f"Specular Reflectivity {method}")
            axs[0].plot(self.q, unumpy.std_devs(self.r), "--", label=f"{method}")
        axs[0].set_ylabel(r"Uncertianties $\sigma_R$")

        plt.xlabel(r"q $[\AA]$")

        for i in range(len(axs)):
            axs[i].legend()
            axs[i].set_yscale("log")
        plt.legend()
        plt.show()


#
#
#


class RawData(XRR):
    """Raw data collected from file dialog"""

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.image_data = None
        self.header_data = None
        self.q = None
        self.energies = None
        self.sample_theta = None
        self.beam_current = None
        self.header_data = None
        self._fits_loader()

    def _fits_loader(self) -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory.

        Computing the xrr profile requires Beamline Energy, Sample Theta, and Beam Current
        Files are opened with astropy and those values are extracted from the header data
        The images are collected from the image data

        """
        file_list = sorted(glob.glob(os.path.join(self.directory, "*.fits")))

        arrays = [
            [
                fits.getheader(f, 0)["Beamline Energy"],
                fits.getheader(f, 0)["Sample Theta"],
                fits.getheader(f, 0)["Beam Current"],
            ]
            for f in file_list
        ]

        self.image_data = np.squeeze(
            np.array([[fits.getdata(f, 2) for f in file_list]])
        )
        self.header_data = np.column_stack(arrays)

        self.energies, self.sample_theta, self.beam_current = self.header_data.T
        self.q = scattering_vector(self.energies, self.sample_theta)


#
#
#


class Images(RawData):
    """2D Image Data"""

    def __init__(self, height=10):
        self.bright_spots = []
        self.dark_spots = []
        self.images = self.image_data
        self.bright_sum = None
        self.dark_sum = None

        self._height = height
        self._beam_spots = []
        self._roi_generator()

    def _roi_generator(self):
        """internal function to find the location of the beam spot on each frame"""
        last_roi = None
        for number, image in enumerate(self.image_data):
            if last_roi is not None and not is_valid_index(image, last_roi):
                roi = last_roi
                raise Exception(
                    f"The beam has drifted more than {self._height} pixels away from its last spot, or the beam has been lost.\n This occurred at frame {number}, we recommend using check_spot(scan_number = {number}) to resolve this"
                )

            else:
                max_idx = np.unravel_index(image.argmax(), image.shape)
                _min = np.array(max_idx) - self.height
                _max = np.array(max_idx) + self.height

                correction = np.zeros(2, dtype=np.int64)
                if np.any(_min < 0):
                    loc = np.where(_min < 0)
                    correction[loc] = -1 * _min[loc]
                if np.any(_max > image.shape[0]):
                    loc = np.where(_max > image.shape[0])
                    correction[loc] = image.shape[0] - _max[loc] - 1

                new_idx = tuple(map(lambda x, y: np.int64(x + y), max_idx, correction))
                assert is_valid_index(image, new_idx)

                roi = (
                    slice(new_idx[0] - height, new_idx[0] + self.height + 1),
                    slice(new_idx[1] - height, new_idx[1] + self.height + 1),
                )

            last_roi = roi
            self.beam_spots.append(new_idx)
            self.bright_spots.append(image[roi])
            self.dark_spots.append(np.flip(image)[roi])
        self.bright_sum = self.bright_spots.sum(axes=(2, 1))
        self.dark_sum = self.dark_spots.sum(axes=(2, 1))

    def check_spot(self, scan_number):
        """external method for checking a scan, and a height"""
        rect_bright = plt.Rectangle(
            self.beam_spots[scan_number],
            self._height,
            self._height,
            edgecolor="red",
            facecolor="None",
        )
        dark_spot = (
            self.images[scan_number].size[0] - self.beam_spots[scan_number][0] - 1,
            self.images[scan_number].size[1] - self.beam_spots[scan_number][1] - 1,
        )
        rect_dark = plt.Rectangle(
            dark_spot, self._height, self._height, edgecolor="red", facecolor="None"
        )

        signal_to_noise = self.bright_sum / self.dark_sum

        s = []
        s.append(f"Scan Number: {scan_number}")
        s.append(f"Beam Energy: {self.energies[scan_number]}")
        s.append(f"Beam Current: {self.beam_current}")
        s.append(f"Sample Theta: {self.sample_theta}")
        s.append("\n")
        s.append(f"Scattering Vector q: {self.q}")
        s.append(f"Bright Spot Intensity: {self.bright_sum}")
        s.append(f"Dark Spot Intensity: {self.dark_sum}")
        s.append(f"Absolute Signal: {self.bright_sum-self.dark_sum}")
        s.append(f"Signal to Noise Ratio: {signal_to_noise}")
        s.append(f"Beam Center: {self.beam_spots[scan_number]}")

        kwargs = {"xticks": [], "yticks": []}
        args = {norm: LogNorm(), cmap: "terrain"}

        fig, ax = plt.subplots(1, 4, subplot_kw=kwargs, figsize=(12, 12))
        ax[0].imshow(image, *args)
        ax[1].imshow(image - dark_sum, *args)

        ax[0].add_patch(rect_dark)
        ax[0].add_patch(rect_bright)

        ax[1].add_patch(rect_dark)
        ax[1].add_patch(rect_bright)

        ax[2].imshow(self.bright_spots[scan_number], *args)
        ax[3].imshow(self.dark_spots[scan_number], *args)

        plt.show()

        return "\n".join(s)


#
#
#


class Reflectivity(Images):
    """stitched and loaded q and r"""

    def __init__(self):
        super().__init__()
        self.r = None
        self.r_err = None

    def _data_reduction(self):
        """Internal Function for reducing 2d image data into a 1d reflectivity curve"""

        self.r = (self.bright_sum - self.dark_sum) / self.beam_current
        self.r_err = np.sqrt(self.bright_sum + self.dark_sum) / self.beam_current

        assert self.q.size == self.r.size

    def _normalize(self):
        """Internal Function for normalizing reflectivity data"""

        izero_count = np.count_nonzero(self.q == 0)

        izero = self.r[: izero_count - 1].mean() if izero_count > 0 else 1
        izero_err = self.r[: izero_count - 1].std() if izero_count > 0 else 0

        self.r_err = (self.r_err / self.r) + (izero_err / izero)
        self.r = self.r[izero_count:] / izero
        self.r_err = self.r_err / self.r
        self.q = self.q[izero_count:]

        assert self.r.size == self.q.size

    def _find_q_series(self):
        """Internal function for locating stitch points in the reflectivity data"""
        ind = np.where(np.diff(self.q) < 0)[0] + 1

        q_split = np.split(self.q, ind)
        r_split = np.split(self.r, ind)
        self._image_split = np.split(self.images, ind)

        stitch_points = min([len(sub) for sub in q_split]) + 1

        self.q_split = [sub for sub in q_split if len(sub) > stitch_points]
        self.r_split = [sub for sub in r_split if len(sub) > stitch_points]

    def _stitch_q_series(self):
        """Internal function for stitching q-series's
        > Additionally uses first r to update stats of an individual q - series"""
        _r_ratios = []
        _r_err_ratios = []
        for i, (sub_q, sub_r) in enumerate(zip(self.q_split, self.r_split)):
            if i == len(self.q_split) - 1:
                pass
            else:
                ratio = np.array([])  # keep track of where the sections overlap
                for j, q in enumerate(sub_q):
                    stitch_indices = np.where(
                        np.isclose(self.q_split[i + 1], q, rtol=tol)
                    )[0]
                    next_r_values = self.r_split[i + 1][stitch_indices]
                    r_value = sub_r[j]
                    ratio = np.append(ratio, r_value / next_r_values)
                ratio = uaverage(ratio)
                self.r_split[i + 1] = self.r_split[i + 1] * ratio
                _ratios.append(ratio)
        self.r = np.concatenate(self.r_split)
        self.q = np.concatenate(self.q_split)
        assert self.r.size == self.q.size


#
#
#


class OneDVisualData(Reflectivity):
    """Contains Visualization Methods of the 1d curves"""


#
#
#


def multi_loader(sort=False):
    parent_dir = file_dialog()
    if sort == True:
        xrr_sorter(parent_dir)

    multi_xrr = []
    for energy in os.listdir(parent_dir):
        full_file_path = os.path.join(parent_dir, energy)
        xrr = XRR(full_file_path)
        xrr.calc_xrr()
        xrr.plot()
        multi_xrr.append(xrr)
    return multi_xrr


def locate_spot(image: np.ndarray, height) -> tuple:
    """
    Locate the bright and dark images of the sample.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of the sample image.

    Returns
    -------
    tuple
        Tuple of the bright and dark image arrays.
    """
    HEIGHT = height  # hard coded dimensions of the spot on the image

    # Get the indices of the maximum and minimum values in the image
    max_idx = np.unravel_index(image.argmax(), image.shape)
    _min = np.array(max_idx) - HEIGHT
    _max = np.array(max_idx) + HEIGHT

    correction = np.zeros(2, dtype=np.int64)
    if np.any(_min < 0):
        loc = np.where(_min < 0)
        correction[loc] = -1 * _min[loc]
    if np.any(_max > image.shape[0]):
        loc = np.where(_max > image.shape[0])
        correction[loc] = image.shape[0] - _max[loc] - 1
    else:
        pass
    new_idx = tuple(map(lambda x, y: np.int64(x + y), max_idx, correction))
    assert is_valid_index(image, new_idx)

    roi = (
        slice(new_idx[0] - HEIGHT, new_idx[0] + HEIGHT + 1),
        slice(new_idx[1] - HEIGHT, new_idx[1] + HEIGHT + 1),
    )

    # Define the regions of interest for the bright and dark spots
    # need to check if spot is within HEIGHT of the edges.
    # If it is, move increase/decrease

    # Extract the bright and dark spots from the image using the ROIs
    u_light = image[roi]
    u_dark = np.flip(image)[roi]
    # if np.any(correction) != 0:
    #     raise Exception("We found the image where the beam runs off!")

    return u_light, u_dark


if __name__ == "__main__":
    # multi_loader()
    dir = file_dialog()
    xrr1 = XRR(dir)
    xrr1.calc_xrr()
    xrr1.plot()
