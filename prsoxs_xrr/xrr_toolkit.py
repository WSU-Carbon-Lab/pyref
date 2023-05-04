import os
import numpy as np
from sympy import true
from uncertainties import unumpy, ufloat
from pathlib import Path
from tkinter import filedialog
from tkinter import *
from typing import Type

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
    if len(arr) == 0:
        return True
    i, j = index
    if i < 0 or i >= arr.shape[0]:
        return False
    if j < 0 or j >= arr.shape[1]:
        return False
    return True


def uaverage(uarray, axis=None, *arge, **kwargs):
    _w = 1 / (unumpy.std_devs(uarray) ** 2)
    return np.average(unumpy.nominal_values(uarray), axis=axis, weights=_w)


if __name__ == "__main__":
    A = np.array([1, 2, 3, 4])
    B = np.array([1, 2, 3, 4])
    C = unumpy.uarray(A, B)
    D = C[1:]
    print(D)
