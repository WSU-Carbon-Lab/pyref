"""Main module."""

from array import array
import os
from pathlib import PureWindowsPath
import typing
import abc
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat

from xrr_toolkit import scattering_vector
from xrr_reduction import reduce


class XRR:
    def __init__(self, Directory) -> None:
        self.Directory = Directory

        # values from loading
        self.meta_data = []
        self.Images: list = []
        self.Thetas: list = []
        self.Energies: list = []
        self.Currents: list = []
        self.Q: list = []

        # data reduction variables
        self.Edgetrim = (5, 5)
        self.Darkside = "LHS"
        self.reduced_data = []

        # stitch points
        self.stitch_subsets = []
        self._scale_factors = []

    def Reflectivity(self):
        raise NotImplementedError(
            "Needs implementation for showing results in a useful format"
        )

    def reduction(self):
        self.reduced_data = reduce(self.meta_d)

    def loader(self):
        self.Energies, self.Currents, self.Thetas, self.Q, self.Images = load_data(
            self.Directory
        )


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
    files: list[str] = [f"{Directory}/{filename}" for filename in os.listdir(Directory)]
    Energies = []
    Currents = []
    Thetas = []
    Images = []
    for file in files:
        with fits.open(file) as hdul:
            Energy = hdul[0].header["Beamline Energy"]
            Theta = hdul[0].header["Sample Theta"]
            Current = hdul[0].header["Beam Current"]
            Image = hdul[2].data
        Energies.append(Energy)
        Thetas.append(Theta)
        Currents.append(Current)
        Images.append(Image)
    Q = scattering_vector(np.array(Energies), np.array(Thetas))
    return (
        np.array(Energies),
        np.array(Currents),
        np.array(Thetas),
        np.array(Q),
        np.array(Images),
    )


if __name__ == "__main__":
    dir = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    XRR(dir).loader()
    print(XRR(dir).reduction())
