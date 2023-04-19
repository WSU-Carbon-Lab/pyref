"""Main module."""

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

    def __init__(self):
        self.directory = None
        self.images = []
        self.energies = []
        self.sample_theta = []
        self.beam_current = []
        self.q = []
        self.r = []
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None

    def load(self, directory: str, error_method="shot", stitch_color="yes") -> None:
        """Load X-ray reflectometry data from FITS files in the specified directory."""

        self.directory = directory
        file_list = [f for f in os.scandir(self.directory) if f.name.endswith(".fits")]

        for file in file_list:
            with fits.open(file.path) as hdul:
                header = hdul[0].header
                energy = header["Beamline Energy"]
                sample_theta = header["Sample Theta"]
                beam_current = header["Beam Current"]
                image_data = hdul[2].data

                self.energies = np.append(self.energies, energy)
                self.sample_theta = np.append(self.sample_theta, sample_theta)
                self.beam_current = np.append(self.beam_current, beam_current)
                self.images = np.append(self.images, image_data)

        self.q = scattering_vector(self.energies, self.sample_theta)
        self.q, self.r = reduce(self.q, self.beam_current, self.images)

    def plot(self) -> None:
        """Plot the X-ray reflectometry data and a reference curve."""

        R = unumpy.nominal_values(self.r)
        R_err = unumpy.std_devs(self.r)

        thommas = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tests",
                "TestData",
                "test.csv",
            ),
            skiprows=1,
            delimiter=",",
            usecols=(1, 2, 3),
        )

        plt.errorbar(self.q, R, yerr=R_err)
        plt.errorbar(thommas[:, 0], thommas[:, 1], yerr=thommas[:, 2])
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    dir = f"{os.getcwd()}\\tests\\TestData\\Sorted\\282.5"
    refl = XRR()
    refl.load(dir)
    refl.plot()
