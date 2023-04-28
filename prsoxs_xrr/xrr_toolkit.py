import os
import numpy as np
from uncertainties import unumpy, ufloat
from pathlib import Path
from tkinter import filedialog
from tkinter import *

c = 299_792_458 * 10**10  # \AA s-2
ħ = 6.582_119_569 * 10 ** (-16)  # eV s


def file_dialog():
    root = Tk()
    root.withdraw()
    directory = Path(filedialog.askdirectory())
    return directory


def scattering_vector(energy, theta):
    global c, ħ
    k = energy / (ħ * c)
    Q = 2 * k * np.sin(np.radians(theta))
    return Q


def is_valid_index(arr, index):
    i, j = index
    if i < 0 or i >= arr.shape[0]:
        return False
    if j < 0 or j >= arr.shape[1]:
        return False
    return True


def uaverage(uarray: np.ndarray) -> ufloat:
    """
    Implementation of weighted average over the input function of ufloats

    Parameters
    ----------
    uarray : np.ndarray
        input array of affine floats (ufloats)

    Returns
    -------
    ufloat
        weighted average of array values
    """
    if uarray.dtype.char == "0":
        nominal_values = unumpy.nominal_values(uarray)
        std_devs = unumpy.std_devs(uarray)

        top = nominal_values / std_devs**2
        bot = 1 / (std_devs**2)

        wavg = ufloat(np.sum(top) / np.sum(bot), np.sqrt(2 / np.sum(bot)))
    else:
        wavg = uarray.mean()
    return wavg
