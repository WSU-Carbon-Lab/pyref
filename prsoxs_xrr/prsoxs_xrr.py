"""Main module."""

from array import array
import os
from pathlib import PureWindowsPath
from typing import *
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat

from xrr_toolkit import scattering_vector
from xrr_reduction import reduce, normalized_reduce


class XRR:
    def __init__(self, Directory) -> None:
        self.Directory: PureWindowsPath("Data Directory") = Directory

        # values from loading
        self.Images: list = []
        self.Thetas: list = []
        self.Energies: list = []
        self.Q = []
        self.Energies, self.Thetas, self.Q, self.Images = load_data(Directory)

        # data reduction variables
        self.Edgetrim = (5, 5)
        self.Darkside = "LHS"
        self.reduced_data = []
        self.reduced_data = normalized_reduce(
            self.Images, self.Q, self.Darkside, self.Edgetrim
        )

        # stitch points
        self._stitch_points = []
        self._scale_factors = []

    def Reflectivity(self):
        raise NotImplementedError(
            "Needs implementation for showing results in a useful format"
        )

    def plot(self, *args, **kwargs):
        refl = self.reduced_data
        plt.errorbar(
            refl["Q"],
            unumpy.nominal_values(refl["Intensity"]),
            unumpy.std_devs(refl["Intensity"]),
            fmt=".",
            *args,
            **kwargs,
        )
        plt.yscale("log")
        plt.title(f"Energy = {np.average(self.Energies):.2g}")
        plt.show()
        plt.xlabel("Q")
        plt.ylabel("R")


def load_data(Directory):
    """
    Parses every .fits file given in ``files`` and returns the meta and image data

    Returns
    -------
    images : list
        List of each image file associated with the .fits
    meta : pd.Dataframe
        pandas dataframe composed of all meta data for each image

    """
    files = [f"{Directory}/{filename}" for filename in os.listdir(Directory)]
    Energies = []
    Thetas = []
    Images = []
    for file in files:
        with fits.open(file) as hdul:
            Energy = hdul[0].header["Beamline Energy"]
            Theta = hdul[0].header["Sample Theta"]
            Image = hdul[2].data
        Energies.append(Energy)
        Thetas.append(Theta)
        Images.append(Image)
    Q = scattering_vector(np.array(Energies), np.array(Thetas))
    return np.array(Energies), np.array(Thetas), np.array(Q), np.array(Images)


if __name__ == "__main__":
    dir = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    XRR(dir).plot()
