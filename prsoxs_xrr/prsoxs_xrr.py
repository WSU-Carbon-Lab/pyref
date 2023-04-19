"""Main module."""

from cProfile import label
import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat

from xrr_toolkit import scattering_vector
from xrr_reduction import reduce


class XRR:
    def __init__(self) -> None:
        self.directory = None

        # values from loading
        self.images: np.ndarray = []
        self.energies: np.ndarray = []
        self.sample_theta: np.ndarray = []
        self.beam_current: np.ndarray = []

        # data reduction variables
        self.q: np.ndarray = []
        self.r: np.ndarray = []

        # total results
        self.xrr: pd.DataFrame = None

        # test methods
        self.std_err = None
        self.shot_err = None

    def load(self, directory, error_method="shot", stitch_color="yes"):
        self.directory = directory
        file_list: list = [f for f in os.listdir(self.directory) if f.endswith(".fits")]

        # loop over all fits files in the directory and extract information
        for file in file_list:
            with fits.open(os.path.join(self.directory, file)) as hdul:
                header = hdul[0].header
                energy = header["Beamline Energy"]
                sample_theta = header["Sample Theta"]
                beam_current = header["Beam Current"]
                image_data = hdul[2].data

                # Append the extracted information to the respective arrays
                self.energies.append(energy)
                self.sample_theta.append(sample_theta)
                self.beam_current.append(beam_current)
                self.images.append(image_data)
        # convert lists to np arrays
        self.energies = np.array(self.energies)
        self.sample_theta = np.array(self.sample_theta)
        self.beam_current = np.array(self.beam_current)
        self.images = np.array(self.images)

        # convert thetas to q's
        self.q = scattering_vector(self.energies, self.sample_theta)
        self.q, self.r = reduce(self.q, self.beam_current, self.images)

    def plot(self) -> None:
        R = unumpy.nominal_values(self.r)
        R_err = unumpy.std_devs(self.r)

        thommas = np.loadtxt(
            f"{os.getcwd()}/tests/TestData/test.csv",
            skiprows=1,
            delimiter=",",
            usecols=(1, 2, 3),
        )

        plt.errorbar(self.q, R, yerr=R_err)
        plt.errorbar(thommas[:, 0], thommas[:, 1], yerr=thommas[:, 2])
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    dir = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    refl = XRR()
    refl.load(dir)
    refl.plot()
