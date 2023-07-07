import numpy as np
import tkinter as tk
import scipy.ndimage as sci
from concurrent.futures import ThreadPoolExecutor


class ImageProcs:
    @staticmethod
    def findMaximum(imageArr: np.ndarray) -> tuple[np.intp, ...]:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        flatIdx = np.argmax(imageArr)
        maxIdx = np.unravel_index(flatIdx, imageArr.shape)
        return maxIdx

    @staticmethod
    def medianFilter(imageArr: np.ndarray, size: int = 3) -> np.ndarray:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

        return sci.median_filter(imageArr, size=size)  # type: ignore

    @staticmethod
    def removeEdge(imageArr: np.ndarray, n: int = 3) -> np.ndarray:
        if not isinstance(imageArr, np.ndarray):
            imageArr = np.ndarray(imageArr)

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
