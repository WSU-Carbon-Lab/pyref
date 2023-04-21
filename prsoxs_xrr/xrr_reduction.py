import numpy as np
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

    bright_sum = bright_spots.sum(axis=(2, 1))
    dark_sum = dark_spots.sum(axis=(2, 1))

    if error_method == "std":
        dark_std = np.array([np.std(np.ravel(u)) for u in dark_spots])
        i_bright = unumpy.uarray(bright_sum, 0)
        i_dark = unumpy.uarray(dark_sum, dark_std)
    elif error_method == "shot":
        i_bright = unumpy.uarray(bright_sum, np.sqrt(bright_sum))
        i_dark = unumpy.uarray(dark_sum, np.sqrt(dark_sum))

    intensity = (i_bright - i_dark) / currents
    R = np.squeeze(intensity)
    Q, R = normalize(Q, R)
    Q, R = stitch_arrays(Q, R)
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

    q_sub = np.squeeze([sub for sub in subsets_Q if len(sub) > 1])
    r_sub = np.squeeze([sub for sub in subsets_R if len(sub) > 1])
    # We could assume that every set of angles has the same number of stitch points but this is in general not correct,
    # we want to be more robust so instead we compute the number of stitch points for each set of angles in the dataset,
    # if they all have the same number of stitch pints then we will reduce this to a scalar funciton

    for i in range(0, np.shape(q_sub)[0] - 1):
        curr_q_sub = q_sub[i]
        curr_r_sub = r_sub[i]
        next_r_sub = r_sub[i + 1]
        next_q_sub = q_sub[i + 1]
        ratio = []
        for j, q in enumerate(curr_q_sub):
            if q in next_q_sub:
                index = np.squeeze(np.where(next_q_sub == q))
                if index.size == 1:
                    ratio.append(
                        min(
                            curr_r_sub[j] / next_r_sub[index],
                            next_r_sub[index] / curr_r_sub[j],
                        )
                    )
                else:
                    ratio.append(
                        min(
                            curr_r_sub[j] / next_r_sub[index[0]],
                            next_r_sub[index[0]] / curr_r_sub[j],
                        )
                    )
                    ratio.append(
                        min(
                            curr_r_sub[j] / next_r_sub[index[1]],
                            next_r_sub[index[1]] / curr_r_sub[j],
                        )
                    )

            else:
                pass
        ratio = np.squeeze(ratio)
        scale_factor = uaverage(np.squeeze(ratio))
        r_sub[i + 1] = next_r_sub * scale_factor

    return np.concatenate(r_sub), np.concatenate(r_sub)


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


def test_subsets(test_case):
    import matplotlib.pyplot as plt

    q = test_case["q"]
    r = test_case["r"]
    correct_results = test_case["correct_r"]
    correct_scales = test_case["correct_scales"]
    qs, rs = stitch_arrays(q, r)
    print(f"q = {q}\n")
    print(f"r = {r}\n")
    print(f"Func output q = {qs}\n")
    print(f"Func output r = {rs}\n")
    print(f"Correct output r = {correct_results}\n")
    for _q, _r in zip(qs, rs):
        plt.plot(_q, unumpy.nominal_values(_r))
    plt.plot(q, unumpy.nominal_values(r))
    plt.show()


# test stitch factors
subset_test_1 = {
    "q": np.asarray([1, 2, 3, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4, 5, 6]),
    "r": np.asarray([6, 5, 4, 10, 10, 8, 6, 16, 16, 12, 8, 24, 24, 16, 8]),
    "correct_r": np.flip([1, 2, 3, 2, 2, 3, 4, 3, 4, 5, 4, 5, 6]),
    "correct_scales": [1, 2, 3, 4],
}

subset_test_2 = {
    "q": np.asarray([1, 2, 3, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4, 5, 6]),
    "r": unumpy.uarray(
        np.asarray([6, 5, 4, 10, 10, 8, 6, 16, 16, 12, 8, 24, 24, 16, 8]),
        np.sqrt(np.asarray([6, 5, 4, 10, 10, 8, 6, 16, 16, 12, 8, 24, 24, 16, 8])),
    ),
    "correct_r": np.flip([1, 2, 3, 2, 2, 3, 4, 3, 4, 5, 4, 5, 6]),
    "correct_scales": [1, 2, 3, 4],
}


if __name__ == "__main__":
    test_subsets(subset_test_2)
