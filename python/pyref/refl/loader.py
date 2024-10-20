"""Simple interfacing module for reducing reflectometry data."""

from collections.abc import Generator

import numpy as np
import polars as pl
import pyref_rs as rs
import skimage
from numba import njit, prange
from result import Ok
from skimage import measure

from pyref.masking import InteractiveImageMasker


class Loader:
    """
    Loader class to load RSoXR data from beamline 11.0.1.2 at the ALS.

    Parameters
    ----------
    path : str
        Path to the experimental directory

    Attributes
    ----------
    path: str
        Path to the directory
    name: str
        Name of the sample - collected from the directory name

    mask: np.ndarray(bool)
        Mask for the data

    dynamic_range: float
        Dynamic range of the detector

    data: pl.DataFrame
        Data loaded from the directory
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = path.split("/")[-1]
        self._raw: pl.DataFrame = Ok(rs.py_read_experiment(path, "xrr")).unwrap_err(
            "Failed to read experiment"
        )

    # =====================/ Properties /=====================

    def __repr__(self) -> str:
        """Return the representation of the Loader object."""
        return self.raw.__repr__()

    def __str__(self) -> str:
        """Return the string representation of the Loader object."""
        return self.raw.__str__()

    @property
    def raw(self) -> pl.DataFrame:
        """Returns the raw data."""
        self.raw = rs.py_simple_update(self._raw, self.path)

    @property
    def mask(self) -> None | np.ndarray:
        """Mask for the data."""
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    def draw_mask(self):
        """Draw the mask on the data."""
        masker = InteractiveImageMasker(self.data.to_numpy())
        self.mask = masker.get_mask()

    # =====================/ Image Processing /=====================

    def img(self, img_col: str) -> Generator[np.ndarray, None, None]:
        """
        Image iterator.

        Parameters
        ----------
        img_col : str
            The name of the column containing image data.

        Yields
        ------
        np.ndarray
            The image data as a numpy array.
        """
        for img in self.data[img_col]:
            shape = int(np.sqrt(len(img)))
            yield rs.py_get_image(img, [shape, shape])

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Raw Image Iterator.

        Yields
        ------
        np.ndarray
            The image data as a numpy array.
        """
        yield from self.img("Raw")

    @property
    def beamspot(self, image, radius=5) -> tuple[int, int]:
        """Locate the beam in the image."""
        img = skimage.filters.gaussian(image, sigma=radius)
        beam_spot = find_max_index(img)
        if beam_spot[0] == 0 or beam_spot[1] == 0 or beam_spot[0] == img.shape[0] - 1:
            # Use edge detection to find the beam
            elevation_map = skimage.filters.sobel(img)
            segmentation = skimage.segmentation.felzenszwalb(
                elevation_map, min_size=2 * radius
            )
            # plot contours of the segmentation on the image
            beam_spot = find_beam_from_contours(segmentation)
        return beam_spot  # Should be a tuple

    def reduce_image(self, image: np.ndarray, radius: int = 5) -> list[float, float]:
        """
        Locate the beam in the image.

        TODO: Implement this function in rust

        Parameters
        ----------
        image : np.ndarray
            The image data as a numpy array.
        radius : int
            The approximate radius of the beam. Used to convolve the "beam" with the
            image in an attempt to flatten objects smaller than the beam.

        Returns
        -------
        list[int, int]
            The specular reflectance of the image and its uncertainty.
        """
        img = apply_mask(image, self.mask)
        img = skimage.filters.gaussian(img, sigma=radius)
        beam_spot = find_max_index(img)
        if beam_spot[0] == 0 or beam_spot[1] == 0 or beam_spot[0] == img.shape[0] - 1:
            # use sobel edge detection to find the beam
            elevation_map = skimage.filters.sobel(img)
            segmentation = skimage.segmentation.felzenszwalb(elevation_map, min_size=10)
            beam_spot = find_beam_from_contours(segmentation)


# =====================/ Numba Functions /=====================
@njit(cache=True, nogil=True)
def find_max_index(arr):
    """Find the index of the maximum value in a 2D array."""
    max_value = arr[0, 0]
    x = 0
    y = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max_value:
                max_value = arr[i, j]
                x = i
                y = j
    return (x, y)  # Return a tuple instead of a list


@njit(cache=True, nogil=True, parallel=True)
def spec_reflectance(masked, beam_spot, box_size):
    """Calculate the specular reflectance of the image."""
    # Get the coordinates
    beam_x = beam_spot[0]
    beam_y = beam_spot[1]

    # Define the ROI boundaries, ensuring they are within image bounds
    roi_start = max(0, beam_x - box_size)
    roi_end = min(masked.shape[0], beam_x + box_size + 1)
    roj_start = max(0, beam_y - box_size)
    roj_end = min(masked.shape[1], beam_y + box_size + 1)

    # Initialize sums and counts
    direct_beam_sum = 0.0
    direct_beam_count = 0
    bg_weighted_sum = 0.0
    bg_weight_total = 0.0

    # Iterate over all rows
    for i in prange(masked.shape[0]):
        # Iterate over all columns
        for j in range(masked.shape[1]):
            value = masked[i, j]
            if value == 0:
                continue
            if (roi_start <= i < roi_end) and (roj_start <= j < roj_end):
                # Within the direct beam region
                direct_beam_sum += value
                direct_beam_count += 1
            else:
                # Background region
                weight = 1.0 / value
                bg_weighted_sum += value * weight
                bg_weight_total += weight

    # Calculate the means
    if direct_beam_sum == 0:
        return (0.0, 0.0)
    if bg_weight_total == 0:
        return (direct_beam_sum, 0.0)

    bg_mean = bg_weighted_sum / bg_weight_total
    bg_std = np.sqrt(1.0 / bg_weight_total)

    spec_reflectance_value = direct_beam_sum - (bg_mean * direct_beam_count)
    uncertainty = np.sqrt(direct_beam_count) + bg_std
    return (spec_reflectance_value, uncertainty)  # Return a tuple


@njit(cache=True, nogil=True)
def apply_mask(img, mask=None, edge=10):
    """Apply a mask to an image."""
    if mask is not None:
        return mask_edge(img * mask, edge=edge)
    return mask_edge(img, edge=edge)


@njit(cache=True, nogil=True)
def mask_edge(image, edge=10):
    """Mask the edge of the image."""
    masked_image = np.copy(image)
    masked_image[:edge, :] = 0
    masked_image[-edge:, :] = 0
    masked_image[:, :edge] = 0
    masked_image[:, -edge:] = 0
    return masked_image


@njit(cache=True, nogil=True)
def find_beam_from_contours(contours: np.ndarray) -> tuple[int, int]:
    """Find the beam from a list of contours."""
    min_area = np.inf
    min_contour = None
    for contour in contours:
        labeled_contour = measure.label(contour)
        props = measure.regionprops(labeled_contour)
        if not props:
            continue
        area = props[0].area
        if area < min_area:
            min_area = area
            min_contour = contour
    if min_contour is None:
        return 0, 0
    # Get the centroid of the contour
    center = np.mean(min_contour, axis=0)
    center = np.round(center).astype(int)
    return tuple(center)


if __name__ == "__main__":
    loader = Loader("/home/hduva/projects/pyref-ccd/test")
    print(loader)
    print(loader.raw)
