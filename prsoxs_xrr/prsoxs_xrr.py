"""Main module."""

from cProfile import label
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy

from xrr_toolkit import scattering_vector, uaverage


class XRR:
    """Class for X-ray reflectometry data analysis."""

    def __init__(self, directory: str):
        self.directory = directory
        self.q = []
        self.r = []
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None
        self.currents = None

    def load(self, error_method="shot", stitch_color="yes") -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory."""

        # Pre-allocate memory for the numpy arrays
        file_list = [f for f in os.scandir(self.directory) if f.name.endswith(".fits")]
        num_files = len(file_list)
        self.energies = np.empty(num_files)
        self.sample_theta = np.empty(num_files)
        self.beam_current = np.empty(num_files)
        self.images = np.empty((num_files, 500, 500))

        for i, file in enumerate(file_list):
            with fits.open(file.path) as hdul:
                header = hdul[0].header
                energy = header["Beamline Energy"]
                sample_theta = header["Sample Theta"]
                beam_current = header["Beam Current"]
                image_data = hdul[2].data

                # Store data in the pre-allocated numpy arrays
                self.energies[i] = energy
                self.sample_theta[i] = sample_theta
                self.beam_current[i] = beam_current
                self.images[i] = image_data

        self.q = scattering_vector(self.energies, self.sample_theta)
        if stitch_color == "yes":
            self.currents = 1
        else:
            self.currents = self.beam_current.mean()

    def reduce(self, error_method="shot") -> None:
        bright_spots, dark_spots = zip(*[locate_spot(image) for image in self.images])
        bright_spots = np.stack(bright_spots)
        dark_spots = np.stack(dark_spots)

        bright_sum = bright_spots.sum(axis=(2, 1))
        dark_sum = dark_spots.sum(axis=(2, 1))

        dark_std = np.array([np.std(np.ravel(u)) for u in dark_spots])

        intensity = bright_sum - dark_sum

        self.std_err = dark_std / self.q.size
        self.shot_err = np.sqrt(bright_sum + dark_sum)
        if error_method == "shot" or error_method == None:
            self.q, self.r = normalize(self.q, unumpy.uarray(intensity, self.shot_err))
        if error_method == "std":
            self.q, self.r = normalize(self.q, unumpy.uarray(intensity, self.std_err))

    def stitch(self):
        self.q, self.r = stitch_arrays(self.q, self.r)

    def plot(self) -> None:
        """Plot the X-ray reflectometry data and a reference curve."""

        R = unumpy.nominal_values(self.r)
        R_err = unumpy.std_devs(self.r)

        thommas = np.loadtxt(
            f"{os.getcwd()}\\tests\\TestData\\test.csv ",
            skiprows=1,
            delimiter=",",
            usecols=(1, 2, 3),
        )

        plt.errorbar(self.q, R, R_err)
        plt.errorbar(thommas[:, 0], thommas[:, 1], yerr=thommas[:, 2])
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


def normalize(Q: np.ndarray, R: np.ndarray) -> tuple:
    """
    Normalization
    """
    # find the cutoff for i_zero points
    izero_count = np.count_nonzero(Q == 0)

    izero = uaverage(R[: izero_count - 1]) if izero_count else ufloat(1, 0)

    R = R[izero_count:] / izero
    Q = Q[izero_count:]

    return Q, R


def stitch_arrays(Q: np.ndarray, R: np.ndarray, tol: float = 1e-6) -> tuple:
    """
    Stitch reflectivity curve
    """
    split_points = np.where(np.diff(Q) < 0)[0] + 1

    subsets_Q = np.split(Q, split_points)
    subsets_R = np.split(R, split_points)

    q_sub = [sub for sub in subsets_Q if len(sub) > 3]
    r_sub = [sub for sub in subsets_R if len(sub) > 3]

    for i in range(0, np.shape(q_sub)[0] - 1):
        curr_q_sub = q_sub[i]
        curr_r_sub = r_sub[i]
        next_r_sub = r_sub[i + 1]
        next_q_sub = q_sub[i + 1]
        ratio = []

        # Create a boolean mask indicating where elements of curr_q_sub appear in next_q_sub within the given tolerance
        mask = np.zeros((len(next_q_sub), len(curr_q_sub)), dtype=bool)
        for j in range(len(next_q_sub)):
            for k in range(len(curr_q_sub)):
                mask[j, k] = np.isclose(
                    next_q_sub[j], curr_q_sub[k], rtol=tol, atol=tol
                )

        # Calculate ratios using broadcasting
        ratios = np.minimum.outer(curr_r_sub, next_r_sub[mask]) / np.maximum.outer(
            curr_r_sub, next_r_sub[mask]
        )

        # Add ratios to list and flatten
        ratio.append(ratios.flatten())

        # Scale next_r_sub by mean of ratios along axis 0
        scale_factor = uaverage(np.array(ratio), axis=0)
        r_sub[i + 1] = next_r_sub * scale_factor

    return np.concatenate(q_sub), np.concatenate(r_sub)


def test_stitchs():
    Rs = refl.r
    Qs = refl.q
    for i, sub in enumerate(Qs):
        plt.plot(sub, unumpy.nominal_values(Rs[i]))
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    dir = f"{os.getcwd()}\\tests\\TestData\\Sorted\\282.5"
    refl = XRR(dir)
    refl.load()
    refl.reduce()
    refl.stitch()
    print(refl.r)
