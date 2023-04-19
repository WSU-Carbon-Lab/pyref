import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat

from xrr_toolkit import uaverage


def reduce(
    Q: np.ndarray, currents: np.ndarray, images: np.ndarray, error_method: str = "shot"
) -> np.ndarray:
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
        elif error_method == "shot":
            i_bright = ufloat(Bright, np.sqrt(Bright))
            i_dark = ufloat(Dark, np.sqrt(Dark))
        int = (i_bright - i_dark) / currents[i]
        R.append(int)

    Q, R = normalize(Q, np.array(R))
    return Q, R


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


def normalize(Q, R):
    """
    Normalization
    """
    # find the cutoff for i_zero points
    izero_count = np.count_nonzero(Q == 0)

    if izero_count == 0:
        izero = ufloat(1, 0)
    else:
        izero = uaverage(R[: izero_count - 1])

    R = R[izero_count:] / izero
    Q = Q[izero_count:]
    return Q, R


def stitch_arrays(Q, R):
    """
    stitch reflectivity curve
    """
    split_points = np.where(np.diff(Q) < 0)[0] + 1

    subsets_Q = np.split(Q, split_points)
    subsets_R = np.split(R, split_points)

    subsets_Q = [sub for sub in subsets_Q if len(sub) > 1]
    subsets_R = [sub for sub in subsets_R if len(sub) > 1]

    # find intersection between subsets
    intersections = []
    for i in range(0, len(subsets_Q) - 1):
        intersection = intersection_with_tolerance(
            subsets_Q[i], subsets_Q[i + 1], assume_unique=False
        )
        intersections.append(intersection)
    return intersections, subsets_Q, subsets_R


def intersection_with_tolerance(arr1, arr2, tol=10 ** (-7), *args, **kwargs):
    """
    Find the intersection between two arrays to a given tolerance.
    """
    arr1_rounded = np.round(arr1 / tol) * tol
    arr2_rounded = np.round(arr2 / tol) * tol
    return np.intersect1d(arr1_rounded, arr2_rounded, *args, **kwargs)


if __name__ == "__main__":
    q = np.array([1, 2.00000001, 3, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4, 5, 6])
    r = np.array([10, 20, 30, 20, 20, 30, 40, 30, 30, 40, 50, 40, 40, 50, 60])
    intersections, subset_q, subset_r = stitch_arrays(q, r)
    rpoints = stitch(q, r)
    print(q)
    print(r)
    print(subset_q)
    print(subset_r)
    print(intersections)
    print(rpoints)
