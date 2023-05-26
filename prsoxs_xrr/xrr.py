"""Main module."""

import copy
import glob
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from astropy.io import fits
from torch import NoneType
from uncertainties import unumpy as unp
from xrr_toolkit import *
import numpy.typing as npt


class XRR:
    """
    Main class for processing xrr data
    """

    def __init__(self, directory, mask=None, *args, **kwargs):
        self.directory = directory
        self.raw_data = RawData(directory, mask=mask, *args, **kwargs)
        self.images = Images(directory, mask=mask, *args, **kwargs)
        self.refl = Reflectivity(directory, *args, **kwargs)
        # Method applications to save data in readable
        self._mask = mask

    def check_spot(self, spot_number, ylims=None, xlims=None):
        self.images.check_spot(spot_number + self.refl._izero_count)
        self.refl.highlight(spot_number, ylims=ylims, xlims=xlims)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask
        self.images = Images(self.directory, self._mask)
        self.refl = Reflectivity(self.directory)


#
#
#


class RawData:
    """Raw data collected from file dialog"""

    def __init__(self, directory, *args, **kwargs):
        # inheritance
        self.directory = directory

        # Constructed properties
        self.image_data: npt.ArrayLike = None
        self.header_data: npt.ArrayLike = None
        self.q: npt.ArrayLike = None
        self.energies: npt.ArrayLike = None
        self.sample_theta: npt.ArrayLike = None
        self.beam_current: npt.ArrayLike = None
        self.header_data: npt.ArrayLike = None

        # object constructor
        self._fits_loader()

    def _fits_loader(self) -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory.

        Computing the xrr profile requires Beamline Energy, Sample Theta, and Beam Current
        Files are opened with astropy and those values are extracted from the header data
        The images are collected from the image data

        """
        self.file_list = sorted(glob.glob(os.path.join(self.directory, "*.fits")))
        self.scan_name = self.file_list[0].split("-")[0].split("\\")[-1]

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

    def save(self):
        file_name = self.directory.name
        scan_info = {
            "Scan ID": self.scan_name,
            "Energy": self.directory.name,
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
    """2D Image Data"""

    def __init__(
        self, directory, mask=None, height=10, ignore_drift=True, *args, **kwargs
    ):
        super().__init__(directory, *args, **kwargs)
        self.bright_spots = []
        self.dark_spots = []
        self.bright_sum = None
        self.dark_sum = None
        if type(mask) is type(None):
            self.mask = None
            self.images = self.image_data
        elif isinstance(mask, np.ndarray) is True:
            self.mask = mask
            self.images = np.squeeze(
                [np.multiply(image, mask) for image in self.image_data]
            )
        else:
            raise TypeError("Mask must be a single numpy array")

        self._ignore_drift = ignore_drift
        self._height = height
        self._beam_spots = []
        self._background_spots = []
        self._roi_generator()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height
        self._roi_generator()

    def _roi_generator(self):
        """internal function to find the location of the beam spot on each frame"""
        self.reduced_roi = []
        for number, image in enumerate(self.images):
            max_idx = np.unravel_index(image.argmax(), image.shape)
            _min = np.array(max_idx) - self._height
            _max = np.array(max_idx) + self._height

            correction = np.zeros(2, dtype=np.int64)
            if np.any(_min < 0):
                loc = np.where(_min < 0)
                correction[loc] = -1 * _min[loc]
            if np.any(_max > image.shape[0]):
                loc = np.where(_max > image.shape[0])
                correction[loc] = image.shape[0] - _max[loc] - 1

            new_idx = tuple(map(lambda x, y: np.int64(x + y), max_idx, correction))
            new_idx_dark = (image.shape[0] - new_idx[0], image.shape[1] - new_idx[1])
            assert is_valid_index(image, new_idx)

            roi = [
                (
                    slice(new_idx[0] - self._height, new_idx[0] + self._height + 1),
                    slice(new_idx[1] - self._height, new_idx[1] + self._height + 1),
                ),
                (
                    slice(
                        new_idx_dark[0] - self._height,
                        new_idx_dark[0] + self._height + 1,
                    ),
                    slice(
                        new_idx_dark[1] - self._height,
                        new_idx_dark[1] + self._height + 1,
                    ),
                ),
            ]
            self.reduced_roi.append(roi)
            self._beam_spots.append((new_idx[0], new_idx[1]))
            self._background_spots.append((new_idx_dark[0], new_idx_dark[1]))

            self.bright_spots.append(image[roi[0]])
            self.dark_spots.append(image[roi[1]])

        self.bright_sum = np.array([np.sum(image) for image in self.bright_spots])
        self.dark_sum = np.array([np.sum(image) for image in self.dark_spots])

    def _show_scan_info(self, scan_number):
        """Build an info dump string that is printed"""

        signal_to_noise = self.bright_sum / self.dark_sum

        signal = (
            self.bright_sum[scan_number] - self.dark_sum[scan_number]
        ) / self.beam_current[scan_number]

        s = []
        s.append(f"Sample: {self.scan_name}")
        s.append(f"Scan Number: {scan_number}")
        s.append(f"Beam Energy: {self.energies[scan_number]:.2g}")
        s.append(f"Polarization: {int(self.polarization[scan_number] - 100)}")
        s.append(f"Beam Current: {self.beam_current[scan_number]:.2g}")
        s.append(f"Sample Theta: {self.sample_theta[scan_number]:.2g}")
        s.append(f"Higher Order Suppressor: {self.hos[scan_number]:.2g}")
        s.append("\n")
        s.append(f"Scattering Vector q: {self.q[scan_number]:.4g}")
        s.append(f"Bright Spot Intensity: {self.bright_sum[scan_number]:.4g}")
        s.append(f"Dark Spot Intensity: {self.dark_sum[scan_number]:.4g}")
        s.append(f"Absolute Signal: {signal:.4g}")
        s.append(f"Signal to Noise Ratio: {signal_to_noise[scan_number]:.4g}")
        s.append(f"Beam Center: {self._beam_spots[scan_number]}")
        s.append("\n")
        print("\n".join(s))

    def generate_mask(self, scan_number):
        self.mask = np.zeros(self.images[scan_number].shape)
        self.mask[self.reduced_roi[scan_number][0]] = 1
        self.mask[self.reduced_roi[scan_number][1]] = 1
        return self.mask

    def check_spot(self, scan_number):
        """external method for checking a scan, and a height"""
        self._show_scan_info(scan_number)

        background_sub = np.maximum(
            (self.images[scan_number] - self.dark_spots[scan_number].mean())
            / self.beam_current[scan_number],
            np.zeros(self.images[scan_number].shape),
        )

        style_kws = {
            "subplots": {"xticks": [], "yticks": []},
            "courser": {"colors": "blue", "lw": 0.8, "ls": "--"},
            "images": {"cmap": "terrain", "norm": colors.LogNorm()},
            "horizontal": {"xmin": 0, "xmax": self.images[scan_number].shape[0] - 1},
            "vertical": {"ymin": 0, "ymax": self.images[scan_number].shape[1] - 1},
        }

        anchor_points = {
            "bright": (
                self._beam_spots[scan_number][1] - self._height - 1,
                self._beam_spots[scan_number][0] - self._height - 1,
            ),
            "dark": (
                self._background_spots[scan_number][1] - self._height - 1,
                self._background_spots[scan_number][0] - self._height - 1,
            ),
        }

        rect_bright = plt.Rectangle(
            anchor_points["bright"],
            2 * self._height + 1,
            2 * self._height + 1,
            edgecolor="blue",
            facecolor="None",
        )

        rect_dark = plt.Rectangle(
            anchor_points["dark"],
            2 * self._height + 1,
            2 * self._height + 1,
            edgecolor="black",
            facecolor="None",
        )

        """Build the return plot"""

        fig, ax = plt.subplots(1, 4, subplot_kw=style_kws["subplots"], figsize=(12, 12))
        axes = ["Raw", "Background Sub", "Bright Spot", "Dark Spot"]
        for a, label in zip(ax, axes):
            a.set_xlabel(label)

        ax[0].imshow(self.images[scan_number], **style_kws["images"])

        ax[1].imshow(background_sub, **style_kws["images"])

        ax[2].imshow(self.bright_spots[scan_number], **style_kws["images"])

        ax[3].imshow(self.dark_spots[scan_number], **style_kws["images"])

        ax[0].add_patch(rect_dark)
        ax[0].add_patch(rect_bright)
        ax[0].hlines(
            self._beam_spots[scan_number][0],
            **style_kws["horizontal"],
            **style_kws["courser"],
        )
        ax[0].vlines(
            self._beam_spots[scan_number][1],
            **style_kws["vertical"],
            **style_kws["courser"],
        )
        plt.show()


#
#
#


class Reflectivity(Images):
    """stitched and loaded q and r"""

    def __init__(self, directory, *args, **kwargs):
        super().__init__(directory, *args, **kwargs)
        self._data_reduction()
        self._normalize()
        self._find_q_series()
        self._stitch_q_series()

    def _data_reduction(self):
        """Internal Function for reducing 2d image data into a 1d reflectivity curve"""

        self.r = unp.uarray(
            (self.bright_sum - self.dark_sum) / self.beam_current,
            np.sqrt(self.bright_sum + self.dark_sum) / self.beam_current,
        )

        assert self.q.size == self.r.size

    def _normalize(self):
        """Internal Function for normalizing reflectivity data"""

        self._izero_count = np.count_nonzero(self.q == 0)

        izero = uaverage(
            self.r[: self._izero_count - 1] if self._izero_count > 0 else 1
        )
        izero_err = np.std(unp.std_devs(self.r[: self._izero_count - 1]))
        self.izero = ufloat(izero, izero_err)

        self.r = self.r[self._izero_count :] / self.izero
        self.q = self.q[self._izero_count :]

        assert self.r.size == self.q.size

    def _find_q_series(self, qtol=1e-3):
        """Internal function for locating stitch points in the reflectivity data"""

        ind = np.where(np.diff(self.q) < -qtol)[0] + 1

        self.q_split = np.split(self.q, ind)
        self.r_split = np.split(self.r, ind)

    def _stitch_q_series(self, stol=1e-1):
        """internal Function for stitching"""
        self._ratios = []
        self._overlap = []
        for i, (sub_q, sub_r) in enumerate(zip(self.q_split, self.r_split)):
            if i == len(self.q_split) - 1:
                pass
            else:
                ratio = np.array([])  # keep track of where the sections overlap
                for j, q in enumerate(sub_q):
                    stitch_indices = np.where(
                        np.isclose(self.q_split[i + 1], q, rtol=stol)
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

    def plot(self, ylims: tuple = None, xlims: tuple = None):
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
        self, highlight, ylims: tuple = None, xlims: tuple = None, *args, **kwargs
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
    xrr1.apply_mask()
    xrr1.check_spot()
