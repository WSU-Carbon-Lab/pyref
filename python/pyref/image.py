"""Image processing functions."""

import numpy as np
import polars as pl
import skimage
from numba import njit, prange
from skimage import measure


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


@njit(cache=True, nogil=True, fastmath=True)
def spec_reflectance(masked, beam_spot, box_size):
    """Calculate the specular reflectance from a masked image."""
    beam_x = beam_spot[0]
    beam_y = beam_spot[1]

    # Define ROI boundaries
    roi_start = max(0, beam_x - box_size)
    roi_end = min(masked.shape[0], beam_x + box_size + 1)
    roj_start = max(0, beam_y - box_size)
    roj_end = min(masked.shape[1], beam_y + box_size + 1)

    # Initialize sums and counts
    direct_beam_sum = 0.0
    direct_beam_count = 0
    direct_beam_weight = 0.0
    bg_weighted_sum = 0.0
    bg_weight_total = 0.0

    # Iterate over all rows and columns
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            value = masked[i, j]
            if value == 0:
                continue

            weight = 1.0 / value
            if (roi_start <= i < roi_end) and (roj_start <= j < roj_end):
                direct_beam_sum += value
                direct_beam_count += 1
                direct_beam_weight += weight
            else:
                bg_weighted_sum += value * weight
                bg_weight_total += weight

    # Calculate the means
    if direct_beam_sum == 0:
        return 0.0, 0.0
    if bg_weight_total == 0:
        return direct_beam_sum, 0.0

    bg_mean = bg_weighted_sum / bg_weight_total

    spec_reflectance_value = direct_beam_sum - (bg_mean * direct_beam_count)
    uncertainty = np.sqrt(direct_beam_count + direct_beam_count / bg_weight_total)

    # # Check for overlap in the error bars
    # #           [-----x------]   <-- Reflectance
    # #    Background -->   [-----x------]
    # if bg_mean + bg_std / 2 > refl_mean - refl_std / 2:
    #     return None, uncertainty

    # Check if the reflectance is negative
    if spec_reflectance_value - uncertainty < 0:
        return None, uncertainty
    return spec_reflectance_value, uncertainty


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


def on_edge(beam_spot, img_shape, roi):
    """Check if the beam spot is on the edge of the image."""
    return (
        beam_spot[0] == roi
        or beam_spot[1] == roi
        or beam_spot[0] == img_shape[0] - roi
        or beam_spot[1] == img_shape[1] - roi
    )


def beamspot(image: np.ndarray, roi: int, radius=10) -> tuple[int, int]:
    """Locate the beam in the image."""
    # convert the image to the correct shape
    img = skimage.filters.gaussian(image, sigma=radius)
    beam_spot = find_max_index(img)
    if on_edge(beam_spot, img.shape, roi):
        # Use edge detection to find the beam
        elevation_map = skimage.filters.sobel(img)
        segmentation = skimage.segmentation.felzenszwalb(
            elevation_map, min_size=roi // 2
        )
        # plot contours of the segmentation on the image
        beam_spot = find_beam_from_contours(segmentation)
    return beam_spot  # Should be a tuple


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def reduce_masked(masks, beam_centers, roi):
    """Reduce the masked images."""
    n_images = masks.shape[0]
    refl = np.zeros(n_images)
    refl_err = np.zeros(n_images)
    for i in prange(n_images):
        mask = masks[i]
        beam_center = beam_centers[i]
        r, e = spec_reflectance(mask, beam_center, roi)
        if r is None and i > 0:
            beam_center = beam_centers[i - 1]
            r, e = spec_reflectance(mask, beam_center, roi)
        if r is None:
            r = float("inf")
            e = e
        refl[i] = r
        refl_err[i] = e
    return refl, refl_err


def pre_process_all(imgs: np.ndarray, roi: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask all the images in the DataFrame."""
    masked = np.zeros_like(imgs)
    beamspots = np.zeros((imgs.shape[0], 2), dtype=np.int32)
    for i in prange(imgs.shape[0]):
        masked[i] = apply_mask(imgs[i])
        beamspots[i] = beamspot(masked[i], roi)
    return masked, beamspots


def locate_beams(df: pl.DataFrame, roi: int) -> tuple[np.ndarray, np.ndarray]:
    """Locate the beams in the images."""
    imgs = df.select("Raw").to_numpy()[:, 0]
    shapes = df.select("Raw Shape").to_numpy()[:, 0]

    reshaped_imgs = np.array(
        [
            np.reshape(img, shape[::-1])[::-1, :]
            for img, shape in zip(imgs, shapes, strict=False)
        ]
    )
    return pre_process_all(reshaped_imgs, roi)


def reduce_masked_data(lzf: pl.DataFrame, roi: int) -> pl.LazyFrame:
    """Reduce the masked data."""
    masked_images, beam_centers = locate_beams(lzf, roi)

    # Stack the masked images into a 3D array
    masked_images_array = np.stack(masked_images) # type: ignore

    # Stack the beam centers into a 2D array
    beam_centers_array = np.vstack(beam_centers) # type: ignore

    # Call the reduce_masked function
    refl, refl_err = reduce_masked(masked_images_array, beam_centers_array, roi)
    if np.any(refl <= 0) or np.any(refl_err <= 0):
        err = f"""Reflectance or error cannot be negative\n
                refl: {refl}\n refl_err: {refl_err}"""
        raise ValueError(err)

    lzf: pl.LazyFrame = lzf.lazy().with_columns(  # type: ignore
        (pl.lit(refl) / pl.col("EXPOSURE [s]")).alias("I [arb. un.]"),
        (pl.lit(refl_err) / pl.col("EXPOSURE [s]")).alias("δI [arb. un.]"),
    )
    return lzf # type: ignore
