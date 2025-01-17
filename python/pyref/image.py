"""Image processing functions."""

import numpy as np
import polars as pl
import skimage
from numba import njit, prange, jit
from skimage import measure
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
    masked: np.ndarray, beam_spot: tuple[int], box_size: int, edge_trim: int
) -> tuple[int, int]:
    """Calculate the specular reflectance from a masked image."""
    beam_x = beam_spot[0]
    beam_y = beam_spot[1]

    # Define ROI boundaries
    roi_start = max(0, beam_x - box_size)
    roi_end = min(masked.shape[0], beam_x + box_size + 1)
    roj_start = max(0, beam_y - box_size)
    roj_end = min(masked.shape[1], beam_y + box_size + 1)

    # Initialize sums and counts
    db_sum = 0
    db_count = 0
    bg_sum = 0
    bg_count = 0

    # Iterate over all rows and columns
    for i in range(masked.shape[1]):
        if i <= edge_trim or i >= masked.shape[1] - edge_trim:
            continue
        for j in range(masked.shape[0]):
            value = masked[i, j]
            if value == 0:
                continue

            if j <= edge_trim or j >= masked.shape[0] - edge_trim:
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

    bg_sum *= db_count / bg_count
    return db_sum, bg_sum


@njit(cache=True, nogil=True)
def reduction1(
    masked: np.ndarray, beam_spot: tuple[int], box_size: int, edge_trim: int
) -> tuple[int, int]:
    # Direct beam intensity
    db_slice1 = slice(beam_spot[0] - box_size // 2, beam_spot[0] + box_size)
    db_slice2 = slice(beam_spot[1] - box_size // 2, beam_spot[1] + box_size)
    db_slice = (db_slice1, db_slice2)
    db = np.sum(masked[db_slice])

    # Background intensity
    side = "rhs" if beam_spot[1] < masked.shape[1] // 2 else "lhs"
    bg_spot = (
        (beam_spot[0], edge_trim)
        if side == "rhs"
        else (beam_spot[0], masked.shape[1] - edge_trim)
    )
    bg_slice1 = slice(bg_spot[0] - box_size // 2, bg_spot[0] + box_size)
    bg_slice2 = slice(bg_spot[1] - box_size // 2, bg_spot[1] + box_size)
    bg_slice = (bg_slice1, bg_slice2)
    bg = np.sum(masked[bg_slice])

    return db, bg


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
def reduce_masked(
    masks, beam_centers, roi, edge_trim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the masked images."""
    n_images = masks.shape[0]
    db = np.zeros(n_images, dtype=np.uint64)
    bg = np.zeros(n_images, dtype=np.uint64)
    for i in prange(n_images):
        mask = masks[i]
        beam_center = beam_centers[i]
        db[i], bg[i] = reduction(mask, beam_center, roi, edge_trim)
    return db, bg


def pre_process_all(imgs: np.ndarray, roi: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask all the images in the DataFrame."""
    masked = np.zeros_like(imgs)
    beamspots = np.zeros((imgs.shape[0], 2), dtype=np.int32)
    for i in prange(imgs.shape[0]):
        zinged = dezinger_image(imgs[i])
        masked[i] = apply_mask(zinged)
        beamspots[i] = beamspot(masked[i], roi)
    return masked, beamspots


def locate_beams(df: pl.DataFrame, roi: int) -> tuple[np.ndarray, np.ndarray]:
    """Locate the beams in the images."""
    imgs = df.select("Raw").to_numpy()[:, 0]
    shapes = df.select("Raw Shape").to_numpy()[:, 0]

    reshaped_imgs = np.array(
        [np.reshape(img, shape)[::-1] for img, shape in zip(imgs, shapes, strict=False)]
    )
    return pre_process_all(reshaped_imgs, roi)


def reduce_masked_data(
    lzf: pl.DataFrame, roi: int, edge_trim: int, shutter_offeset: float = 0.00389278
) -> pl.LazyFrame:
    """Reduce the masked data."""
    masked_images, beam_centers = locate_beams(lzf, roi)

    # Stack the masked images into a 3D array
    masked_images_array = np.stack(masked_images)  # type: ignore

    # Stack the beam centers into a 2D array
    beam_centers_array = np.vstack(beam_centers)  # type: ignore

    # Call the reduce_masked function
    direct_beam, background = reduce_masked(
        masked_images_array, beam_centers_array, roi, edge_trim
    )
    lzf: pl.LazyFrame = lzf.lazy().with_columns(  # type: ignore
        (
            pl.lit(direct_beam - background)
            / (
                (pl.col("EXPOSURE [s]") + pl.lit(shutter_offeset))
                * pl.col("Beam Current [mA]")
            )
        ).alias("I [arb. un.]"),
        (
            pl.lit(direct_beam + background).sqrt()
            / (
                (pl.col("EXPOSURE [s]") + pl.lit(shutter_offeset))
                * pl.col("Beam Current [mA]")
            )
        ).alias("δI [arb. un.]"),
    )
    return lzf  # type: ignore


def dezinger_image(image: np.ndarray, threshold=10, size=3) -> np.ndarray:
    """Dezinger an image."""
    # set false values to zero
    image[image < 0] = 0
    med_result = median_filter(image, size=size)  # Apply Median Filter to image

    diff_image = image / np.abs(
        med_result
    )  # Calculate Ratio of each pixel to compared to a threshold
    # Repopulate image by removing pixels that exceed the threshold -- From Jan Ilavsky's IGOR implementation.
    output = image * np.greater(threshold, diff_image).astype(
        int
    ) + med_result * np.greater(
        diff_image, threshold
    )  #
    return output  # Return dezingered image and averaged image
