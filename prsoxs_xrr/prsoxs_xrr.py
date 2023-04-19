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
    def __init__(self):
        self.directory = None
        self.images = np.empty(0)
        self.energies = np.empty(0)
        self.sample_theta = np.empty(0)
        self.beam_current = np.empty(0)
        self.q = np.empty(0)
        self.r = np.empty(0)
        self.xrr = pd.DataFrame()
        self.std_err = None
        self.shot_err = None

    def load(self, directory, error_method="shot", stitch_color="yes"):
        self.directory = directory
        file_list = [f for f in os.listdir(self.directory) if f.endswith(".fits")]

        for file in file_list:
            with fits.open(os.path.join(self.directory, file)) as hdul:
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

    def plot(self):
        R = unumpy.nominal_values(self.r)
        R_err = unumpy.std_devs(self.r)

        thommas = np.loadtxt(
            os.path.join(os.getcwd(), "tests", "TestData", "test.csv"),
            skiprows=1,
            delimiter=",",
            usecols=(1, 2, 3),
        )

        plt.errorbar(self.q, R, yerr=R_err)
        plt.errorbar(thommas[:, 0], thommas[:, 1], yerr=thommas[:, 2])
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    dir = os.path.join(os.getcwd(), "tests", "TestData", "Sorted", "282.5")
    refl = XRR()
    refl.load(dir)
    refl.plot()
