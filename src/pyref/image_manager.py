from typing import assert_type
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage as sci


class ImageProcs:
    @staticmethod
    def findMaximum(imageArr: np.ndarray) -> tuple[np.intp, ...]:
        _imageInputErr(imageArr)

        flatIdx = np.argmax(imageArr)
        maxIdx = np.unravel_index(flatIdx, imageArr.shape)
        return maxIdx

    @staticmethod
    def medianFilter(imageArr: np.ndarray, size: int = 3) -> np.ndarray:
        _imageInputErr(imageArr)

        return sci.median_filter(imageArr, size=size)  # type: ignore

    @staticmethod
    def removeEdge(imageArr: np.ndarray, n: int = 3) -> np.ndarray:
        (imageArr)

        return imageArr[n:-n, n:-n]

    @staticmethod
    def roiReduction(
        imageArr: np.ndarray, center: tuple[int, int], height: int, width: int
    ) -> np.ndarray:
        assert_type(center, tuple[int, int])
        assert_type(height, int)
        assert_type(width, int)
        cy, cx = center
        mx, my = imageArr.shape

        top = max(cy - height // 2, 0)
        bot = min(cy + height // 2 + height % 2, mx)

        left = max(cx - width // 2, 0)
        right = min(cx + width // 2 + width % 2, mx)

        roi = imageArr[top:bot, left:right]
        return roi

    @staticmethod
    def oppositePoint(
        center: tuple[int, int], array_shape: tuple[int, int]
    ) -> tuple[int, int]:
        assert_type(center, tuple[int, int])
        assert_type(array_shape, tuple[int, int])

        y, x = center
        height, width = array_shape

        opposite_x = width - x - 1
        opposite_y = height - y - 1

        return opposite_y, opposite_x

    @staticmethod
    def applyMask(imageArr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return imageArr[mask]

    @staticmethod
    def sumImage(imageArr: ArrayLike | np.ndarray) -> int:
        return imageArr.sum() #type: ignore


def _imageInputErr(imageArr) -> Exception | None:
    # TODO: Add error handling
    ...