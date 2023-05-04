"""Main module."""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pkg_resources import ensure_directory
from sympy import is_zero_dimensional
from uncertainties import unumpy as unp
import glob
import matplotlib.colors as colors

from xrr_toolkit import *

# os.environ['OMP_NUM_THREADS'] =


class XRR:
    """
    Main class for processing xrr data
    """

    def __init__(self, directory: str, *args, **kwargs):
        self.directory = directory
        self.raw_data = RawData(self.directory, *args, **kwargs)
        self.images = Images(self.directory, *args, **kwargs)
        self.refl = Reflectivity(self.directory, *args, **kwargs)


#
#
#


class RawData:
    """Raw data collected from file dialog"""

    def __init__(self, directory, *args, **kwargs):
        self.directory = directory

        # Constructed properties
        self.image_data = None
        self.header_data = None
        self.q = None
        self.energies = None
        self.sample_theta = None
        self.beam_current = None
        self.header_data = None

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
            ]
            for f in self.file_list
        ]

        self.image_data = np.squeeze(
            np.array([[fits.getdata(f, 2) for f in self.file_list]])
        )
        self.header_data = np.column_stack(arrays)

        self.energies, self.sample_theta, self.beam_current = self.header_data
        self.q = scattering_vector(self.energies, self.sample_theta)

    def save(self, file_name: str):
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

    def __init__(self, directory, height=10, ignore_drift=False, *args, **kwargs):
        super().__init__(directory, *args, **kwargs)
        self.bright_spots = []
        self.dark_spots = []
        self.images = self.image_data
        self.bright_sum = None
        self.dark_sum = None

        self._ignore_drift = ignore_drift
        self._height = height
        self._beam_spots = []
        self._roi_generator()

    def _roi_generator(self):
        """internal function to find the location of the beam spot on each frame"""
        last_roi = []
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
            assert is_valid_index(image, new_idx)

            roi = (
                slice(new_idx[0] - self._height, new_idx[0] + self._height + 1),
                slice(new_idx[1] - self._height, new_idx[1] + self._height + 1),
            )

            if number != 0:
                if (
                    is_valid_index(image[last_roi[number - 1]], new_idx) == False
                    and self._ignore_drift == False
                ):
                    roi = last_roi[number - 1]
                    raise Exception(
                        f"Beam shifts out of last roi indicating beam drift at frame number {number}\n",
                        f"Recomend using check_spot({number}) to determine the nature of this drift \n",
                        f"To continue without this, set ignote_drift = True",
                    )
            last_roi.append(roi)
            self._beam_spots.append(new_idx)
            self.bright_spots.append(image[roi])
            self.dark_spots.append(np.flip(image)[roi])
        self.bright_sum = np.sum(self.bright_spots, axis=(2, 1))
        self.dark_sum = np.sum(self.dark_spots, axis=(2, 1))

    def check_spot(self, scan_number):
        """external method for checking a scan, and a height"""
        rect_bright = plt.Rectangle(
            self._beam_spots[scan_number],
            self._height,
            self._height,
            edgecolor="red",
            facecolor="None",
        )
        dark_spot = (
            self.images[scan_number].shape[0] - self._beam_spots[scan_number][0] - 1,
            self.images[scan_number].shape[1] - self._beam_spots[scan_number][1] - 1,
        )
        rect_dark = plt.Rectangle(
            dark_spot, self._height, self._height, edgecolor="red", facecolor="None"
        )

        signal_to_noise = self.bright_sum / self.dark_sum

        s = []
        s.append(f"Sample: {self.scan_name}")
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
        s.append(f"Beam Center: {self._beam_spots[scan_number]}")

        kwargs = {"xticks": [], "yticks": []}
        args = {"cmap": "terrain"}

        fig, ax = plt.subplots(1, 4, subplot_kw=kwargs, figsize=(12, 12))
        ax[0].imshow(self.images[scan_number], norm=colors.LogNorm(), cmap="terrain")
        ax[1].imshow(
            self.images[scan_number] - self.dark_sum[scan_number],
            norm=colors.LogNorm(),
            cmap="terrain",
        )

        ax[0].add_patch(rect_dark)
        ax[0].add_patch(rect_bright)

        ax[2].imshow(
            self.bright_spots[scan_number], norm=colors.LogNorm(), cmap="terrain"
        )
        ax[3].imshow(
            self.dark_spots[scan_number], norm=colors.LogNorm(), cmap="terrain"
        )

        plt.show()

        return "\n".join(s)


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

        izero_count = np.count_nonzero(self.q == 0)

        izero = uaverage(self.r[: izero_count - 1] if izero_count > 0 else 1)
        izero_err = np.std(unp.std_devs(self.r[: izero_count - 1]))
        izero = ufloat(izero, izero_err)

        self.r = self.r[izero_count:] / izero
        self.q = self.q[izero_count:]

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

    def plot(self, *args, **kwargs):
        plt.yscale("log")
        plt.errorbar(
            self.q,
            unp.nominal_values(self.r),
            unp.std_devs(self.r),
            fmt=".",
            *args,
            **kwargs,
        )
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
    # multi_loader()
    dir = file_dialog()

    xrr = XRR(dir)
    xrr.raw_data.save(dir.name)
    xrr.refl.plot()
    xrr.refl.plot_stitches()
