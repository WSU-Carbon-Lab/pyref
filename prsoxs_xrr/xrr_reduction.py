from enum import IntEnum
from turtle import right
from matplotlib import axes
import numpy as np
import pandas as pd
import typing
import abc
from sympy import is_monotonic
from sympy import is_monotonic
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt

from xrr_toolkit import uaverage


def reduce(meta_data: pd.DataFrame, images: list, edge_cut: int = 5) -> pd.DataFrame:
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

    # unpack meta data
    currents = meta_data["Beam Current"]
    Q = meta_data["Q"]

    # locate beam on each frame
    R = []

    # Integrate bright and dark intensity, background subtract.
    for i, u in enumerate(images):
        bright_spot, dark_spot = locate_spot(u, edge_cut)

        Bright = np.nansum(np.ravel(bright_spot))
        i_bright = ufloat(Bright, np.sqrt(Bright))

        Dark = np.nansum(np.ravel(dark_spot)) / len(~np.isnan(np.ravel(dark_spot)))
        i_dark = ufloat(Dark, np.sqrt(Dark))

        int = (i_bright - i_dark) / currents[i]  # type: ignore
        R.append(int)

    pre_stitch_normal = pd.DataFrame(columns=["Q", "R"])
    pre_stitch_normal["Q"] = Q
    pre_stitch_normal["R"] = R

    pre_stitch = normalization(pre_stitch_normal)

    return pre_stitch


def locate_spot(image: np.ndarray, edge_cut: int) -> tuple:
    HEIGHT = 4  # hard coded dimensions of the spot on the image
    i, j = np.unravel_index(np.ravel(image).argmax(), image.shape)

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

    return np.array(u_light), np.array(u_dark)


def normalization(refl: pd.DataFrame):
    """
    Normalization
    """
    # find the cutoff for i_zero points
    izero_count = refl["Q"].where(refl["Q"] == 0).count()

    if izero_count == 0:
        izero = ufloat(1, 0)
    else:
        izero = uaverage(refl["R"].iloc[:izero_count])

    refl["R"] = refl["R"] / izero  # type: ignore
    refl["R"] = refl["R"].drop(refl["R"].index[:izero_count])
    return refl


def stitching(refl: pd.DataFrame):
    """
    stitch reflectivity curve
    """

    # unpack refl
    Q = refl["Q"].to_numpy()
    R = refl["R"].to_numpy()

    def split_increasing(arr, image):
        """
        Splits the given array `arr` into monotonically decreasing subarrays
        of size at least 2 and first entries being at least `thre`.
        """
        split_points = np.where(np.diff(arr) < 0)[0] + 1

        sub_lists_dom = np.split(arr, split_points)
        sub_lists_im = np.split(image, split_points)

        result_dom = [sub for sub in sub_lists_dom if sub.size > 1]
        result_im = [sub for sub in sub_lists_im if sub.size > 1]

        return result_dom, result_im

    subsets = split_increasing(Q, R)

    return subsets


if __name__ == "__main__":
    refl = pd.DataFrame(columns=["Q", "R"])
    refl["Q"] = [
        0,
        1,
        2,
        3,
        4,
        1,
        1,
        2,
        3,
        4,
        5,
        2,
        2,
        3,
        4,
        5,
        6,
    ]
    refl["R"] = [
        10,
        11,
        12,
        13,
        14,
        11,
        11,
        12,
        13,
        14,
        15,
        12,
        12,
        13,
        14,
        15,
        16,
    ]
    stitches = stitching(refl)

    print(stitches)
