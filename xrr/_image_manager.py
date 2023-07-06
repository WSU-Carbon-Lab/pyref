from typing import Any
from copy import deepcopy
import numpy as np
import tkinter as tk
import scipy.ndimage as sci

from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor


class ImageProcs:
    @staticmethod
    @np.vectorize
    def findMaximum(imageArr: np.ndarray) -> tuple[np.intp, ...]:
        _inputErrChecker(imageArr)

        flatIdx = np.argmax(imageArr)
        maxIdx = np.unravel_index(flatIdx, imageArr.shape)
        return maxIdx

    @staticmethod
    @np.vectorize
    def medianFilter(imageArr: np.ndarray, size: int = 3) -> np.ndarray:
        _inputErrChecker(imageArr)

        return sci.median_filter(imageArr, size=size)  # type: ignore

    @staticmethod
    @np.vectorize
    def removeEdge(imageArr: np.ndarray, n: int = 3) -> np.ndarray:
        _inputErrChecker(imageArr)

        return imageArr[n:-n, n:-n]

    @staticmethod
    @np.vectorize
    def applyMask(imageArr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        _inputErrChecker(imageArr)
        _inputErrChecker(mask)

        imageArr[~mask] = 0
        return imageArr

    @staticmethod
    def generateMask(imageArr, x1, y1, x2, y2):
        mask = np.zeros_like(imageArr, dtype=bool)
        mask[y1:y2, x1:x2]


class ImageDisplay:
    def __init__(self, imageArr) -> None:
        _inputErrChecker(imageArr)

        # Data variables
        self.image = imageArr
        self.mask = np.zeros_like(imageArr, dtype=bool)

        # tkinter envoroment variables
        self.root = None
        self.canvas: tk.Canvas = None  # type:ignore
        self.photo = None
        self.masking = False
        self.dragBox: DragBox = None  # type: ignore

    def show(self, *args: Any, **kwds: Any) -> Any:
        self.root = tk.Tk()
        self.root.title("CCD Image")

        PIL_image = Image.fromarray(ImageProcs.removeEdge(self.image))
        self.photo = ImageTk.PhotoImage(PIL_image)

        self.canvas = tk.Canvas(
            self.root, width=PIL_image.width, height=PIL_image.height
        )
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.onMouseClick)
        self.canvas.bind("<B1-Motion>", self.onMouseDrag)
        self.canvas.bind("<ButtonRelease-1>", self.onMouseRelease)

    def onMouseClick(self, event):
        self.masking = True
        self.dragBox = DragBox(self.canvas, event.x, event.y)

    def onMouseDrag(self, event):
        if self.masking:
            self.dragBox.update(event.x, event.y)

    def onMouseRelease(self, event):
        self.masking = False
        x1, y1, x2, y2 = self.dragBox.getCoords()
        self.mask = ImageProcs.generateMask(self.image, x1, y1, x2, y2)
        self.dragBox.destroy()

    def updateImage(self):
        maskedImage = ImageProcs.applyMask(self.image, self.mask)

        PIL_image = Image.fromarray(maskedImage)
        self.photo = ImageTk.PhotoImage(PIL_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


class DragBox:
    def __init__(self, canvas, xStart, yStart) -> None:
        self.canvas = canvas
        self.xStart = xStart
        self.yStart = yStart

        self.rectangle = self.canvas.create_rectangle(
            xStart, yStart, xStart, yStart, outline="red"
        )

    def update(self, xEnd, yEnd):
        self.canvas.create_rectangle(
            self.xStart, self.yStart, xEnd, yEnd, outline="red"
        )

    def getCoords(self):
        maskBox = self.canvas.coords(self.rectangle)
        return maskBox

    def destroy(self):
        self.canvas.delete(self.rectangle)


def _inputErrChecker(imageArr) -> None:
    if not isinstance(imageArr, np.ndarray):
        imageArr = np.ndarray(imageArr)

    if imageArr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")


image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
image_display = ImageDisplay(image)
image_display.show()
