import os
import numpy as np
from uncertainties import unumpy, ufloat

c = 299_792_458 * 10**10  # \AA s-2
ħ = 6.582_119_569 * 10 ** (-16)  # eV s


def scattering_vector(Energy, Theta):
    k = Energy / (ħ * c)
    Q = 2 * k * np.sin(np.radians(Theta))
    return Q


def uaverage(uarray):
    nominal_values = unumpy.nominal_values(uarray)
    std_devs = unumpy.std_devs(uarray)

    top = nominal_values / std_devs**2
    bot = 1 / (std_devs**2)

    wavg = ufloat(np.sum(top) / np.sum(bot), np.sqrt(2 / np.sum(bot)))
    return wavg
