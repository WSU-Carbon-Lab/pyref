import numpy as np
from uncertainties import ufloat, unumpy

from xrr_toolkit import uaverage


def reduce(
    Q: np.ndarray, currents: np.ndarray, images: np.ndarray, error_method: str = "shot"
) -> np.ndarray:
    R_top = np.nansum(np.ravel(locate_spot(images)[0]), axis=1) - np.nansum(
        np.ravel(locate_spot(images)[1]), axis=1
    )
    R_bottom = np.sqrt(np.nansum(np.ravel(locate_spot(images)[0]), axis=1))
    if error_method == "std":
        R_err = R_bottom / np.sqrt(len(np.ravel(locate_spot(images)[0])))
    else:
        R_err = R_bottom
    R = unumpy.uarray(R_top, R_err) / currents[:, np.newaxis]
    Q, R_norm = normalize(Q, R)
    fancy_image = None
    return Q, R_norm, fancy_image


def locate_spot(image: np.ndarray) -> tuple:
    """
    Locate the bright and dark images of the sample.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of the sample image.

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
    top_d, bottom_d = -(j + 1) - HEIGHT, -(j + 1) + HEIGHT

    u_slice_d = (slice(left_d, right_d), slice(top_d, bottom_d))
    u_dark = image[u_slice_d]

    return np.array(u_light), np.array(u_dark)


def normalize(Q: np.ndarray, R: np.ndarray) -> tuple:
    """
    Normalization
    """
    # find the cutoff for i_zero points
    izero_count = np.count_nonzero(Q == 0)

    izero = uaverage(R[: izero_count - 1]) if izero_count else ufloat(1, 0)

    R = R[izero_count:] / izero
    Q = Q[izero_count:]

    return Q, R


def stitch_arrays(Q: np.ndarray, R: np.ndarray) -> tuple:
    """
    Stitch reflectivity curve
    """
    split_points = np.where(np.diff(Q) < 0)[0] + 1

    subsets_Q = np.split(Q, split_points)
    subsets_R = np.split(R, split_points)

    subsets_Q = [sub for sub in subsets_Q if len(sub) > 1]
    subsets_R = [sub for sub in subsets_R if len(sub) > 1]

    # find intersection between subsets
    intersections = []
    for i in range(len(subsets_Q) - 1):
        intersection = intersection_with_tolerance(subsets_Q[i], subsets_Q[i + 1])
        intersections.append(intersection)

    return intersections, subsets_Q, subsets_R


def intersection_with_tolerance(
    arr1: np.ndarray, arr2: np.ndarray, tol: float = 10 ** (-7)
) -> np.ndarray:
    """
    Find the intersection between two arrays to a given tolerance.
    """
    return arr1[np.isclose(arr1, arr2, rtol=0, atol=tol)]


if __name__ == "__main__":
    from astropy.io import fits
    import os
    import matplotlib.pyplot as plt

    file = (
        f"{os.getcwd()}\\tests\\TestData\\Sorted\\282.5\\ZnPc_P100_E180276-00001.fits"
    )
    image = fits.getdata(file, ext=2)
    u_light, u_dark = locate_spot(image)
    plt.imshow(u_dark)
    plt.show()
