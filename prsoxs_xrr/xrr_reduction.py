from enum import IntEnum
from turtle import right
from matplotlib import axes
import numpy as np
import pandas as pd
import typing
import abc
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from xrr_toolkit import scattering_vector


def reduce(meta_data: pd.DataFrame, image: list, edge_cut: int = 5) -> pd.DataFrame:
    """
    Reduces Images to 1d intensity values for each Q and Current

    Parameters
    ----------
    meta_data : DataFrame
        DataFrame packed as

            meta_data = ["Beam Current", "Sample Theta", "Beamline Energy", "Q"]

    images : list
        list of images collected from fits files

    Returns
    -------
    Reflectivity : DataFrame
        DataFrame packed as

            Reflectivity = ['Q', 'Intensity']

        Intensity : ufloat64
            Computed Intensity and its uncertianty

        Q : float
            scattering vector
    """
    # unpack meta_data

    currents = meta_data["Beam Current"]
    Q = meta_data["Q"]

    # locate Beam and Dark Spot and make areas of integration about them

    bright_spot, dark_spot = locate_spots(image, edge_cut)

    # Integrate bright and dark intensity, background subtract.

    i_bright_n = np.sum(np.ravel(bright_spot)), np.sqrt(np.sum(np.ravel(bright_spot)))
    i_dark_n = np.sum(np.ravel(dark_spot)) / len(np.ravel(dark_spot))

    i_bright = ufloat(i_bright_n, np.sqrt(i_bright_n))
    i_dark = ufloat(i_dark_n, np.sqrt(i_dark_n))

    pre_norm_intensities = (i_bright - i_dark) / currents

    # pack array into desired output format
    #   Reflectivity = [Q, R]

    pre_stitch_normal = pd.DataFrame(columns=["Q", "R"])
    pre_stitch_normal["Q"] = Q
    pre_stitch_normal["R"] = pre_norm_intensities

    # normalize, and stitch

    Reflectivity: pd.DataFrame = stitch(normalize(pre_stitch_normal))

    return Reflectivity


def locate_spots(Images: list, edge_cut: int) -> tuple:
    HEIGHT = 4

    for n, image in enumerate(Images):
        # for each image find brightest spot
        i, j = np.unravel_index(np.max(image), image.shape)

        left = i - HEIGHT
        right = i + HEIGHT
        top = j - HEIGHT
        bottom = j + HEIGHT

        u_slice: tuple[slice, slice] = (slice(left, right), slice(top, bottom))
        u_light = image[u_slice]

        u_dark = image.copy()
        for k in range(len(image[0])):
            for l in range(len(image[1])):
                if top < k < bottom and right < l < left:
                    u_dark[k][l] = np.nan
                else:
                    pass

    return u_light, u_dark


def normalize(pre_norm: pd.DataFrame) -> pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    pre_norm : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # pre_norm = ['Q', 'R']

    num_i_zeros = pre_norm["Q"].where(pre_norm["Q"] == 0).count()
    i_zero = np.sum(pre_norm["R"].where(pre_norm["Q"] == 0)) / num_i_zeros

    pre_norm["R"] = pre_norm["R"] / i_zero

    return pre_norm.drop(pre_norm.index[:num_i_zeros])


def close_to_any(array, a):
    return ~np.isclose(array, a, atol=0.00001).any()


def stitch(data_not_stitched: pd.DataFrame) -> pd.DataFrame | None:
    Q_values = data_not_stitched["Q"]
    prior_qs = []
    INDICATOR_index = []

    for i, q in enumerate(Q_values):
        last_q = Q_values[i - 1]

        current_intensity = data_not_stitched["R"]

        # if q is less than q old, set off indicator to start stitching
        if q < last_q:
            INDICATOR = True
            INDEX = i
            INDICATOR_index.append(i)

        # while the indicator is true stitch
        while INDICATOR == True:
            scale_factors_n = []
            scale_factors_std = []

            # check if there exists a similar near neighbor if not then undo indicator and scale reflectivity data
            if not close_to_any(prior_qs, q):
                weighted_scale_factors = np.average(
                    scale_factors_n, weights=scale_factors_std
                )
                future_data = data_not_stitched["R"].iloc[INDEX:]
                INDICATOR = False

            # if we are at the first indicator check all data before it for stitch points
            if len(INDICATOR_index) == 1:
                # find nearest q
                myList = data_not_stitched["Q"].iloc[:i]
                nearest_q = min(myList, key=lambda x: abs(x - q))

                # determine scale factor
                nearest_q_int = data_not_stitched["R"].where(
                    data_not_stitched["Q"] == nearest_q
                )
                scale_factor = nearest_q_int / current_intensity
                scale_factors_n.append(unumpy.nominal_values(scale_factor))
                scale_factors_std.append(unumpy.std_devs(scale_factor))
            # else only check last set of angles
            else:
                myList = data_not_stitched["Q"].iloc[
                    INDICATOR_index[INDEX - 1] : INDICATOR_index[INDEX]
                ]
                nearest_q = min(myList, key=lambda x: abs(x - q))

                # determine scale factor
                nearest_q_int = data_not_stitched["R"].where(
                    data_not_stitched["Q"] == nearest_q
                )
                scale_factor = nearest_q_int / current_intensity
                scale_factors_n.append(unumpy.nominal_values(scale_factor))
                scale_factors_std.append(unumpy.std_devs(scale_factor))

        prior_qs.append(q)
        return data_not_stitched
