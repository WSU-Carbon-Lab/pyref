from re import S
from typing import assert_type
import numpy as np
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
        _imageInputErr(imageArr)

        return imageArr[n:-n, n:-n]

    @staticmethod
    def rectangleMaskGen(
        shape: tuple[int], center: tuple[int], height: int, width: int
    ) -> np.ndarray:
        assert_type(center, tuple[int])
        assert_type(height, int)
        assert_type(width, int)

        mask = np.ones(shape).astype(bool)
        rowStart = center[0] - width // 2
        rowStop = center[0] + width // 2 + width % 2

        colStart = center[0] - height // 2
        colStop = center[0] + height // 2 + height % 2

        rowSlice = slice(rowStart, rowStop)
        colSlice = slice(colStart, colStop)

        sliceObj = (rowSlice, colSlice)
        mask[sliceObj] = False
        return mask

    @staticmethod
    def applyMask(
        imageArr: np.ndarray, mask: np.ndarray, inverted: bool = False
    ) -> np.ndarray:
        _imageInputErr(imageArr)
        _maskInputErr(mask)

        if not inverted:
            imageArr[mask] = 0
        elif inverted:
            imageArr[~mask] = 0
        else:
            raise ValueError("inverted flag must be a boolean value")

        return imageArr

    @staticmethod
    def sumImage(imageArr: np.ndarray) -> float:
        _imageInputErr(imageArr)

        return imageArr.sum()


def _imageInputErr(imageArr) -> Exception | None:
    if not isinstance(imageArr, np.ndarray):
        imageArr = np.ndarray(imageArr)

    if imageArr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")


def _maskInputErr(maskArr) -> Exception | None:
    if not isinstance(maskArr, np.ndarray):
        maskArr = np.ndarray(maskArr)

    if not (maskArr.dtype == type(bool)):
        if np.all(np.logical_or(maskArr == 0, maskArr == 1) == 0):
            maskArr = maskArr.astype(bool)

        else:
            raise ValueError(
                "Input array must be a boolean array or an array of 0 and 1"
            )
