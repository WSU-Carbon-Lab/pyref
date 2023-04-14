from enum import IntEnum
import numpy as np
import pandas as pd
import typing
import abc
from uncertainties import unumpy, ufloat

from xrr_toolkit import scattering_vector


def reduce(Images, Q, Currents, edge_cut=(5, 5)):
    """
    Reduces Images to 1d intensity values for each Q and Current

    Parameters
    ----------
    Images : array-like
        Array containing images to reduce
    Q : array-like
        Scattering vector for each Image
    Currents : array-like
        Beamline current at the time of each image

    Returns
    -------
    Reflectivity : DataFrame
        Dataframe packed as

            Reflectivity = ['Q', 'Intensity']

        Intensity : ufloat64
            Computed Intensity and its uncertianty

        Q : float
            scattering vector
    """
    # locate Beam and Dark Spot and make areas of integration about them

    bright_spot, dark_spot = locate_spots(Images)

    # Integrate bright and dark intensity, background subtract.

    i_bright = np.sum(bright_spot, axis=1)
    i_dark = np.sum(dark_spot, axis=1)

    pre_norm_intensities = (i_bright - i_dark) / Currents

    # pack array into desired output format
    #   Reflectivity = [Q, R]

    pre_stitch_normal = np.array(Q, pre_norm_intensities)

    # normalize, and stitch

    Reflectivity = stitch(normalize(pre_stitch_normal))

    return Reflectivity


def locate_spots(Images):
    bright_loc = Images.argmax(axis=1)
    return bright_loc


def normalize(data_not_normalized):
    raise NotImplementedError
    return data_not_normalized


def stitch(data_not_stitched):
    raise NotImplementedError
    return data_not_stitched
