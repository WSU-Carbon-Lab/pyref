"""Image processing functions."""

from __future__ import annotations

import numpy as np
import polars as pl
import skimage
from numba import njit
from scipy.ndimage import median_filter


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


@njit(cache=True, nogil=True)
def reduction(
    masked: np.ndarray, beam_spot: tuple[int, int], roi: int
) -> tuple[int, int]:
    """Calculate the specular reflectance from a masked image."""
    beam_x = beam_spot[0]
    beam_y = beam_spot[1]

    # Define ROI boundaries
    roi_start = max(0, beam_x - roi)
    roi_end = min(masked.shape[0], beam_x + roi + 1)
    roj_start = max(0, beam_y - roi)
    roj_end = min(masked.shape[1], beam_y + roi + 1)

    # Initialize sums and counts
    db_sum = 0
    db_count = 0
    bg_sum = 0
    bg_count = 0

    # Iterate over all rows and columns
    for i in range(masked.shape[1]):
        for j in range(masked.shape[0]):
            value = masked[i, j]
            if value == 0:
                continue

            if (roi_start <= i < roi_end) and (roj_start <= j < roj_end):
                db_sum += value
                db_count += 1
            else:
                bg_sum += value
                bg_count += 1

    if bg_count == 0:
        return 0, 0
    if db_sum == 0:
        return 0, 0

    bg_sum *= int(db_count / bg_count)
    return int(db_sum), int(bg_sum)


@njit(cache=True, nogil=True)
def apply_mask(
    img: np.ndarray, mask: np.ndarray | None = None, edge: int = 10
) -> np.ndarray:
    """Apply a mask to an image."""
    if mask is not None:
        return mask_edge(img * mask, edge=edge)
    return mask_edge(img, edge=edge)


@njit(cache=True, nogil=True)
def mask_edge(image: np.ndarray, edge: int) -> np.ndarray:
    """Set the edge values of an image to zero."""
    image[:edge, :] = 0
    image[-edge:, :] = 0
    image[:, :edge] = 0
    image[:, -edge:] = 0
    return image


def find_beam_from_contours(
    img: np.ndarray, segmentation: np.ndarray
) -> tuple[int, int]:
    """Find the beam from a list of contours."""
    cluster = np.argmax(
        [np.sum(img[segmentation == i]) for i in np.unique(segmentation) if i != 0]
    )
    segmentation = segmentation == cluster
    # beamspot is in the center of the cluster
    y, x = np.where(segmentation)
    x = int(np.mean(x))
    y = int(np.mean(y))
    return (x, y)


@njit(cache=True, nogil=True)
def row_by_row_subtraction(image: np.ndarray) -> np.ndarray:
    """Subtract the average of the left or right side of the image."""
    left = image[:, :20]
    right = image[:, -20:]
    if left.sum() < right.sum():
        # subtract row by row the average of the right side
        image = image - right.mean(axis=1)[:, None]
    else:
        image = image - left.mean(axis=1)[:, None]
    return image


def on_edge(beam_spot, img_shape, roi):
    """Check if the beam spot is on the edge of the image."""
    return (
        beam_spot[0] == roi
        or beam_spot[1] == roi
        or beam_spot[0] == img_shape[0] - roi
        or beam_spot[1] == img_shape[1] - roi
    )


def locate_beam(image, roi):
    """Locate the beam in the image."""
    # do a quick and poorly done background subtraction
    left = image[:, :20]
    right = image[:, -20:]
    if left.sum() < right.sum():
        image = image - right.mean(axis=1)[:, None]
    else:
        image = image - left.mean(axis=1)[:, None]

    beam_spot = find_max_index(image)
    if on_edge(beam_spot, image.shape, roi):
        # Use edge detection to find the beam
        u8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
            np.uint8
        )
        elevation_map = skimage.filters.sobel(u8)
        segmentation = skimage.segmentation.felzenszwalb(
            elevation_map, min_size=roi // 2, scale=roi
        )
        beam_spot = find_beam_from_contours(u8, segmentation)
    return beam_spot


def reduce_data(
    image: np.ndarray, mask: np.ndarray, roi: int, radius: int, edge_trim: int
) -> tuple[int, int]:
    """Locate the beam in the image."""
    # convert the image to the correct shape
    image = np.array(image)
    zinged = dezinger_image(image)
    filtered = skimage.filters.gaussian(zinged, sigma=radius)
    masked = apply_mask(filtered, mask=mask, edge=edge_trim)
    beam_spot = locate_beam(masked, roi)
    return reduction(zinged, beam_spot, roi)


def reduce_masked_data(
    df: pl.DataFrame,
    mask: np.ndarray,
    roi: int,
    radius: int,
    edge_trim: int,
    shutter_offeset: float = 0.00389278,
) -> pl.LazyFrame:
    """Reduce the masked data."""
    return (
        df.lazy()
        .with_columns(
            pl.col("Raw")
            .map_elements(
                lambda x: reduce_data(x, mask, roi, radius, edge_trim),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("I [beam], I [background]"),
        )
        .with_columns(
            (
                pl.col("I [beam], I [background]").map_elements(
                    lambda x: x[0] - x[1], return_dtype=pl.Int64
                )
                / (
                    (pl.col("EXPOSURE [s]") + pl.lit(shutter_offeset))
                    * pl.col("Beam Current [mA]")
                )
            ).alias("I [arb. un.]"),
            (
                pl.col("I [beam], I [background]")
                .map_elements(lambda x: x[0] + x[1], return_dtype=pl.Int64)
                .sqrt()
                / (
                    (pl.col("EXPOSURE [s]") + pl.lit(shutter_offeset))
                    * pl.col("Beam Current [mA]")
                )
            ).alias("δI [arb. un.]"),
        )
        .drop(["Raw", "I [beam], I [background]"])
    )


def dezinger_image(image: np.ndarray, threshold=10, size=3) -> np.ndarray:
    """Dezinger an image."""
    # set false values to zero
    image[image < 0] = 0
    med_result = median_filter(image, size=size)  # Apply Median Filter to image

    diff_image = image / np.abs(
        med_result
    )  # Calculate Ratio of each pixel to compared to a threshold
    # Repopulate image by removing pixels that exceed the threshold --
    # From Jan Ilavsky's IGOR implementation.
    output = image * np.greater(threshold, diff_image).astype(
        int
    ) + med_result * np.greater(diff_image, threshold)  #
    return output  # Return dezingered image and averaged image
