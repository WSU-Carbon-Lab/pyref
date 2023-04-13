"""Main module."""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat

import xrr_toolkit


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
            Theta = round(hdul[0].header["CCD Theta"], 2)
            Images.append(hdul[2].data)
        Energies.append(Energy)
        Thetas.append(Theta)
    UsefulData = {"Energy": Energies, "Theta": Thetas}
    Data = pd.DataFrame(UsefulData)
    return Data, Images


def Calculate_Integral_Ranges(Image, edge_trim=(5, 5)) -> tuple[range, range]:
    N_x, N_y = Image.shape
    bright_spot_x, bright_spot_y = np.unravel_index(Image.argmax(), Image.shape)
    temporary_x_range = range(edge_trim[0], N_x - edge_trim[0])
    temporary_y_range = range(edge_trim[1], N_y - edge_trim[1])
    if bright_spot_x in temporary_x_range:
        x_range = temporary_x_range
    elif bright_spot_x < edge_trim[0]:
        x_range = range(0, N_x - edge_trim[0])
    else:
        x_range = range(edge_trim[0], N_x)
    if bright_spot_y in temporary_y_range:
        y_range = temporary_y_range
    elif bright_spot_y < edge_trim[1]:
        y_range = range(0, N_y - edge_trim[1])
    else:
        y_range = range(edge_trim[1], N_y)
    return x_range, y_range


def normalization(Data):
    i_zero_points = Data[Data["Q"] == 0]["Intensity"]
    i_zero_array = i_zero_points.to_numpy()
    Data["Normalized Intensity"] = Data["Intensity"] / np.mean(i_zero_array)
    raise NotImplementedError("Normalization not correctly implemented")


def find_stitch_points(Data):
    direct_beam = Data[Data["Q"] == 0]["Intensity"]
    raise NotImplementedError("Stitch points not implemented")


if __name__ == "__main__":
    dir = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    Data, Images = load_data(dir)
    Refl = data_reduction(Images, Data)
    i_zero = normalization(Refl)
    print(i_zero)
