import numpy as np
import scipy.ndimage as sci
from concurrent.futures import ThreadPoolExecutor


class ImageProcs:
    @staticmethod
    @np.vectorize
    def findMaximum(imageArr: np.ndarray) -> tuple[np.intp, ...]:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        if imageArr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")

        flatIdx = np.argmax(imageArr)
        maxIdx = np.unravel_index(flatIdx, imageArr.shape)
        return maxIdx

    @staticmethod
    @np.vectorize
    def medianFilter(imageArr: np.ndarray, size: int = 3) -> np.ndarray:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        if imageArr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")

        return sci.median_filter(imageArr, size=size)  # type: ignore

    @staticmethod
    @np.vectorize
    def removeEdge(imageArr: np.ndarray, n: int = 3) -> np.ndarray:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        if imageArr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")

        return imageArr[n:-n, n:-n]

    @staticmethod
    @np.vectorize
    def applyMask(imageArr: np.ndarray, mask: slice) -> np.ndarray:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        if imageArr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")
        imageArr[mask] = np.nan
        return imageArr

    @staticmethod
    def generateMask(imageArr: np.ndarray) -> np.ndarray:
        ...
