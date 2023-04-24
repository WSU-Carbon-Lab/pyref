"""Main module."""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy
import glob

from xrr_toolkit import scattering_vector, uaverage


class XRR:
    """
    Main class for processing xrr data
    """

    def __init__(self, directory: str, error_method=None, *args, **kwargs):
        self.directory = directory
        self.q = np.array([], dtype=np.float64)
        self.r = np.array([], dtype=np.float64)
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None

    # define the outcome of the call

    def calc_xrr(self, *args, **kwargs):
        # Load fits and perform data reduction, stitching, and normalization
        self._fits_loader()
        self._data_reduction(*args, **kwargs)
        self._normalize()
        self._stitch(*args, **kwargs)

        # Assign values to xrr dictionary
        self.xrr["Q"] = self.q
        self.xrr["R"] = unumpy.nominal_values(self.r)
        self.xrr["R_err"] = unumpy.std_devs(
            self.r
        )  # call std_devs() to calculate standard deviation of self.r

        # Return xrr dictionary
        return self.xrr

    def _fits_loader(self) -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory."""

        # Get list of .fits files in the directory
        file_list = sorted(glob.glob(os.path.join(self.directory, "*.fits")))

        # Pre-allocate memory for the numpy arrays
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

        # Calculate the scattering vector
        self.q = scattering_vector(self.energies, self.sample_theta)

    def _data_reduction(self, error_method="shot") -> None:
        bright_spots, dark_spots = zip(*[locate_spot(image) for image in self.images])
        bright_spots = np.stack(bright_spots)
        dark_spots = np.stack(dark_spots)

        bright_sum = bright_spots.sum(axis=(2, 1))
        dark_sum = dark_spots.sum(axis=(2, 1))

        dark_std = np.array([np.std(np.ravel(u)) for u in dark_spots])

        r = (bright_sum - dark_sum) / self.beam_current
        self.std_err = dark_std / self.q.size
        self.shot_err = np.sqrt(bright_sum + dark_sum)
        if error_method == "shot":
            self.r = unumpy.uarray(r, self.shot_err)
        elif error_method == "std":
            self.r = unumpy.uarray(r, self.std_err)
        elif error_method == "compare":
            self.r1 = unumpy.uarray(r, self.shot_err)
            self.r2 = unumpy.uarray(r, self.std_err)
        else:
            raise Exception(
                'Choose "shot" for possonian statistics, "std" for gaussian statistics, or "compare" to compare the two methods'
            )

    def _normalize(self):
        """
        Normalization
        """
        # find the cutoff for i_zero points
        izero_count = np.count_nonzero(self.q == 0)

        izero = uaverage(self.r[: izero_count - 1]) if izero_count > 0 else ufloat(1, 0)

        self.r = self.r[izero_count:] / izero
        self.q = self.q[izero_count:]

    def _stitch(self, tol=0.1):
        # split self.q and self.r into sections that are monotonically increasing
        ind = np.where(np.diff(self.q) < 0)[0] + 1

        q_split = np.split(self.q, ind)
        r_split = np.split(self.r, ind)

        stitch_points = min([len(sub) for sub in q_split]) + 1

        self.q_split = [sub for sub in q_split if len(sub) > stitch_points]
        self.r_split = [sub for sub in r_split if len(sub) > stitch_points]

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

    def plot(self) -> None:
        """Plot the X-ray reflectometry data and a reference curve."""
        for i, sub in enumerate(zip(self.q_split, self.r_split)):
            plt.errorbar(
                sub[0],
                unumpy.nominal_values(sub[1]),
                unumpy.std_devs(sub[1]),
                fmt=".",
                label=f"Scan = {i+2}",
            )
        plt.legend()
        plt.yscale("log")
        plt.show()


def locate_spot(image: np.ndarray) -> tuple:
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
    HEIGHT = 4  # hard coded dimensions of the spot on the image

    # Get the indices of the maximum and minimum values in the image
    max_idx = np.unravel_index(image.argmax(), image.shape)

    # Define the regions of interest for the bright and dark spots
    roi = (
        slice(max_idx[0] - HEIGHT, max_idx[0] + HEIGHT + 1),
        slice(max_idx[1] - HEIGHT, max_idx[1] + HEIGHT + 1),
    )

    # Extract the bright and dark spots from the image using the ROIs
    u_light = image[roi]
    u_dark = np.flip(image)[roi]

    return u_light, u_dark


if __name__ == "__main__":
    dir = f"{os.getcwd()}\\tests\\TestData\\Sorted\\282.5"
    xrr = XRR(dir)
    xrr.calc_xrr()
    xrr.plot()
