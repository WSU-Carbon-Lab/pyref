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
    """
    General XRR class; for loading, reducing, and plotting xrr data from a fits file.
    """

    def __init__(self, Directory) -> None:
        self.Directory = Directory

        # values from loading
        self.meta_data = loader(self.Directory)[0]
        self.images = loader(self.Directory)[1]

        # data reduction variables
        self.edge_trim = 5
        self.dark_side = "LHS"
        self.reduced_data: pd.DataFrame = reduce(
            self.meta_data, self.images, self.edge_trim
        )

        # total results
        self.xrr = None

    def show_images(self) -> None:
        pass

    def plot_data(self) -> None:
        q = self.reduced_data["Q"]
        R = unumpy.nominal_values(self.reduced_data["R"])
        R_err = unumpy.std_devs(self.reduced_data["R"])

        thommas = pd.read_csv(f"{os.getcwd()}/tests/TestData/test.csv")

        plt.errorbar(q, R, yerr=R_err)
        # plt.errorbar(thommas["Q"], thommas["R"], yerr=thommas["R_err"])
        plt.yscale("log")
        plt.show()


def loader(dirr):
    files: list[str] = [f"{dirr}/{filename}" for filename in os.listdir(dirr)]

    # init dictionary for meta data, list for images

    temp_meta = {}
    images = []
    meta_data = None

    # generate list of import meta data information

    HEADER_MASTER_LIST = ["Sample Theta", "Beamline Energy", "Beam Current"]

    for i, filename in enumerate(files):
        with fits.open(filename) as hdul:
            header = hdul[0].header  # type: ignore
            for item in header:
                if item in HEADER_MASTER_LIST:
                    temp_meta[item] = header[item]
                else:
                    pass
            images.append(hdul[2].data)  # type: ignore
        if i == 0:
            meta_data = pd.DataFrame(temp_meta, index=[i])
        else:
            meta_data = pd.concat([meta_data, pd.DataFrame(temp_meta, index=[i])])
    meta_data["Q"] = scattering_vector(  # type: ignore
        meta_data["Beamline Energy"], meta_data["Sample Theta"]  # type: ignore
    )
    return [meta_data, images]


if __name__ == "__main__":
    dir = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    refl = XRR(dir)
    refl.plot_data()
