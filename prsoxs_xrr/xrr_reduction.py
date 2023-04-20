import numpy as np
from torch import norm
from uncertainties import ufloat, unumpy

from xrr_toolkit import uaverage


import numpy as np
import uncertainties.unumpy as unumpy


def reduce(
    Q: np.ndarray, currents: np.ndarray, images: np.ndarray, error_method: str = "shot"
) -> np.ndarray:
    bright_spots, dark_spots = zip(*[locate_spot(image) for image in images])
    bright_spots = np.stack(bright_spots)
    dark_spots = np.stack(dark_spots)

    bright_sum = np.sum(bright_spots.sum(axis=2, keepdims=True), axis=1)
    dark_sum = np.sum(dark_spots.sum(axis=2, keepdims=True), axis=1)

    if error_method == "std":
        bright_std = np.std(bright_spots, axis=1) / np.sqrt(
            bright_spots.shape[1] * bright_spots.shape[2]
        )
        dark_std = np.std(dark_spots, axis=1) / np.sqrt(
            dark_spots.shape[1] * dark_spots.shape[2]
        )
        i_bright = unumpy.uarray(bright_sum, bright_std)
        i_dark = unumpy.uarray(dark_sum, dark_std)
    elif error_method == "shot":
        i_bright = unumpy.uarray(bright_sum, np.sqrt(bright_sum))
        i_dark = unumpy.uarray(dark_sum, np.sqrt(dark_sum))

    int = (i_bright - i_dark) / currents[:, np.newaxis]
    R = np.squeeze(int)
    Q, R = normalize(Q, R)
    return Q, R


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

    # Get the indices of the maximum and minimum values in the image
    max_idx = np.unravel_index(image.argmax(), image.shape)

    # Define the regions of interest for the bright and dark spots
    roi = (
        slice(max_idx[0] - HEIGHT, max_idx[0] + HEIGHT + 1),
        slice(max_idx[1] - HEIGHT, max_idx[1] + HEIGHT + 1),
    )

    # Extract the bright and dark spots from the image using the ROIs
    u_light = image[roi]
    u_dark = np.flip(image)[roi]

    return u_light, u_dark


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

    matching_indices = [
        np.where(np.in1d(subsets_Q[i], subsets_Q[i + 1]))
        for i in range(0, len(subsets_Q) - 1)
    ]
    
    stitch_points = 

    return subsets_Q, subsets_R, matching_indices


def test_slices():
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import os

    path = f"{os.getcwd()}/tests/TestData/Sorted/282.5/"
    file_list = [f for f in os.scandir(path) if f.name.endswith(".fits")]
    for file in file_list:
        image = fits.getdata(file, ext=2)
        u_bright, u_dark = locate_spot(image)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image)
        axs[1].imshow(u_bright)
        axs[2].imshow(u_dark)
        plt.show()


def test_subsets():
    q = np.asarray([1, 2, 3, 2, 3, 4, 3, 4, 5])
    r = 1 / q
    rs, qs, indices = stitch_arrays(q, r)
    print(q)
    print(r)
    print(qs)
    print(rs)
    print(indices)


if __name__ == "__main__":
    test_subsets()
