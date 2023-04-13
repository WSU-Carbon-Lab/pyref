import os
import numpy as np

c = 299_792_458 * 10**10  # \AA s-2
ħ = 6.582_119_569 * 10 ** (-16)  # eV s


def scattering_vector(Energy, Theta):
    k = Energy / (ħ * c)
    Q = 2 * k * np.sin(np.radians(Theta))
    return Q
