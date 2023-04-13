import numpy as np
import pandas as pd
from typing import *

from xrr_toolkit import scattering_vector
from uncertainties import unumpy, ufloat


def Calculate_Integral_Ranges(Image, edge_trim) -> tuple[range, range]:
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


def reduce(Images, Q, Darkside, Edgetrim):
    reduced_data = []

    if Darkside == "LHS":
        dark_izero = 0
    elif Darkside == "RHS":
        raise NotImplementedError(
            "Input Darkside must be 'LHS' for current implementation"
        )
    else:
        raise ValueError("Choose either Darkside = 'LHS' or Darkside = 'RHS'")

    for i, image in enumerate(Images):
        Rx, Ry = Calculate_Integral_Ranges(image, Edgetrim)
        sliced_image = image[Rx[0] : Rx[-1]][Ry[0] : Ry[-1]]
        intensity = ufloat(np.sum(sliced_image), np.sqrt(np.sum(sliced_image)))
        reduced_data.append([Q[i], intensity])
    return pd.DataFrame(reduced_data, columns=["Q", "Intensity"])


def normalized_reduce(*args, **kwargs):
    raw_reduced_data = reduce(*args, **kwargs)

    i_zeros = raw_reduced_data["Intensity"].where(raw_reduced_data["Q"] == 0).dropna()
    i_zero = np.sum(i_zeros.to_numpy()) / len(i_zeros.to_numpy())
    reduced_data = raw_reduced_data.copy()
    reduced_data["Intensity"] = reduced_data["Intensity"] / i_zero

    return reduced_data.drop(reduced_data.index[: len(i_zeros.to_numpy()) - 1])


def find_stitch_points(*args, **kwargs):
    normalized_data = normalized_data.copy(*args, **kwargs)
    for Q in normalized_data["Q"]:
        pass
    raise NotImplementedError


def calculate_stitch_factor():
    raise NotImplementedError


def calculate_stitch():
    raise NotImplementedError
