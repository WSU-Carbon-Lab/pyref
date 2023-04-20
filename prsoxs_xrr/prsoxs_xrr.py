"""Main module."""

from cProfile import label
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy

from xrr_toolkit import scattering_vector
from xrr_reduction import reduce


class XRR:
    """Class for X-ray reflectometry data analysis."""

    def __init__(self, directory: str):
        self.directory = directory
        self.q = []
        self.r = []
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None

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
        self.q, self.r = reduce(self.q, self.beam_current, self.images)

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


def test_stitchs():
    Rs = refl.r
    Qs = refl.q
    for i, sub in enumerate(Qs):
        plt.plot(sub, unumpy.std_devs(Rs[i]))
    plt.show()


if __name__ == "__main__":
    dir = f"{os.getcwd()}\\tests\\TestData\\Sorted\\282.5"
    refl = XRR(dir)
    refl.load()
    refl.plot()
