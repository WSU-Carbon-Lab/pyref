import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat

from xrr_toolkit import uaverage


def reduce(
    meta_data: pd.DataFrame, images: list, error_method: str = "shot"
) -> pd.DataFrame:
    """
    Reduce Images to 1d intensity values for each Q and Current.

    Parameters
    ----------
    meta_data : pd.DataFrame
        DataFrame containing "Beam Current", "Sample Theta", "Beamline Energy", "Q".
    images : list
        List of images collected from fits files.
    error_method : str, optional
        The error method to use: "shot" or "std" (default is "shot").

    Returns
    -------
    pd.DataFrame
        DataFrame packed as ["Q", "R"] with "R" as computed intensity and its uncertainty.
    """
    currents, Q = meta_data["Beam Current"], meta_data["Q"]
    R = []

    # Integrate bright and dark intensity, background subtract.
    for i, u in enumerate(images):
        bright_spot, dark_spot = locate_spot(u)
        Bright = np.nansum(np.ravel(bright_spot))
        Dark = np.nansum(np.ravel(dark_spot))
        if error_method == "std":
            Bright_std = np.nanstd(np.ravel(bright_spot)) / np.sqrt(
                len(np.ravel(bright_spot))
            )
            Dark_std = np.nanstd(np.ravel(dark_spot)) / np.sqrt(
                len(np.ravel(dark_spot))
            )
            i_bright = ufloat(Bright, Bright_std)
            i_dark = ufloat(Dark, np.sqrt(Dark))
        else:
            i_bright = ufloat(Bright, np.sqrt(Bright))
            i_dark = ufloat(Dark, np.sqrt(Dark))
        int = (i_bright - i_dark) / currents[i]
        R.append(int)

    pre_stitch_normal = pd.DataFrame({"Q": Q, "R": R})
    pre_stitch = normalization(pre_stitch_normal)
    return pre_stitch


def locate_spot(image: np.ndarray) -> tuple:
    """
    Locate the bright and dark images of the sample.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of the sample image.
    edge_cut : int
        Number of pixels to cut from image edges.

    Returns
    -------
    tuple
        Tuple of the bright and dark image arrays.
    """
    HEIGHT = 4  # hard coded dimensions of the spot on the image
    i, j = np.unravel_index(image.argmax(), image.shape)

    left, right = i - HEIGHT, i + HEIGHT
    top, bottom = j - HEIGHT, j + HEIGHT

    u_slice_l = (slice(left, right), slice(top, bottom))
    u_light = image[u_slice_l]

    left_d, right_d = -(i + 1) - HEIGHT, -(i + 1) + HEIGHT
    top_d, bottom_d = -(i + 1) - HEIGHT, -(i + 1) + HEIGHT

    u_slice_d = (slice(left_d, right_d), slice(top_d, bottom_d))
    u_dark = image[u_slice_d]

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
    from astropy.io import fits
    import os
    from uncertainties import unumpy

    file = f"{os.getcwd()}/tests/TestData/Sorted/282.5"
    refl = loader(file)

    q = refl["Q"]
    R = unumpy.nominal_values(refl["R"])
    R_err = unumpy.std_devs(refl["R"])

    thommas = pd.read_csv(f"{os.getcwd()}/tests/TestData/test.csv")

    plt.errorbar(q, R, yerr=R_err)
    plt.errorbar(thommas["Q"], thommas["R"], yerr=thommas["R_err"])
    plt.yscale("log")
    plt.show()


def locate_spot_old(image: np.ndarray) -> tuple:  # older version
    """
    Locate the bright and dark images of the sample.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of the sample image.
    edge_cut : int
        Number of pixels to cut from image edges.

    Returns
    -------
    tuple
        Tuple of the bright and dark image arrays.
    """
    HEIGHT = 4  # hard coded dimensions of the spot on the image
    i, j = np.unravel_index(image.argmax(), image.shape)

    left, right = i - HEIGHT, i + HEIGHT
    top, bottom = j - HEIGHT, j + HEIGHT

    u_slice = (slice(left, right), slice(top, bottom))
    u_light = image[u_slice]

    u_dark = image.copy()
    for k in range(len(image[0])):
        for l in range(len(image[1])):
            if top < k < bottom and right < l < left:
                u_dark[k][l] = np.nan
            else:
                pass

    return np.array(u_light), np.array(u_dark)
