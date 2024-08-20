import re
import tkinter
import tkinter.dialog
import tkinter.filedialog
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import cv2 as cv
import numpy as np
import pandas as pd
import polars as pl
from astropy.io import fits

# --------------------------------------------------------
# Custom Enums and Structs
# --------------------------------------------------------


class Motor(Enum):
    """Enum for motor values."""

    Energy = ("Beamline Energy",)
    Theta = ("Sample Theta",)
    Current = ("Beam Current",)
    Hos = ("Higher Order Suppressor",)
    Pol = ("EPU Polarization",)
    Exposure = ("EXPOSURE",)


@dataclass
class ImageData:
    """Dataclass for image data."""

    image: np.ndarray = None
    processed: np.ndarray = None
    direct_beam: tuple[int, int] = None

    def read_file(self, hdu: fits.HDUList) -> Self:
        """Read the image data from a FITS file."""
        self.image = hdu[2].data
        self._process_image()
        return self

    def _process_image(self):
        """Process the image using OpenCV."""
        # Convert the image to grayscale
        self.processed = cv.normalize(self.image, None, 0, 255, cv.NORM_MINMAX).astype(
            np.uint8
        )
        # Apply a Gaussian blur to reduce noise
        blurred_image = cv.GaussianBlur(self.processed, (5, 5), 0)

        # Threshold the image to create a binary image
        _, threshold_image = cv.threshold(
            blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        # Find contours in the binary image
        contours, _, _ = cv.findContours(
            threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Iterate through the contours and find the largest contour
        largest_contour = None
        largest_contour_area = 0
        for contour in contours:
            contour_area = cv.contourArea(contour)
            if contour_area > largest_contour_area:
                largest_contour = contour
                largest_contour_area = contour_area

        # Find the centroid of the largest contour
        if largest_contour is not None:
            moments = cv.moments(largest_contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            self.direct_beam = (centroid_x, centroid_y)

        # Create a mask of the largest contour
        mask = np.zeros_like(self.image)
        cv.drawContours(mask, [largest_contour], 0, 255, -1)

        # Apply the mask to the original image
        self.processed = cv.bitwise_and(self.image, self.image, mask=mask)
        return self


@dataclass
class HeaderData:
    """Dataclass for header data."""

    energy: Motor.Energy = None
    theta: Motor.Theta = None
    current: Motor.Current = None
    hos: Motor.Hos = None
    pol: Motor.Pol = None
    exposure: Motor.Exposure = None
    image: ImageData = None

    def __post_init__(self):
        self.image = ImageData()

    def read_file(self, hdu: fits.HDUList) -> Self:
        """Read the header data from a FITS file."""
        self.energy = hdu[0].header["Beamline Energy"]
        self.theta = hdu[0].header["Sample Theta"]
        self.current = hdu[0].header["Beam Current"]
        self.hos = hdu[0].header["Higher Order Suppressor"]
        self.pol = hdu[0].header["EPU Polarization"]
        self.exposure = hdu[0].header["EXPOSURE"]

        self.image.read_file(hdu)

        return self


# --------------------------------------------------------
# Loaders
# --------------------------------------------------------


def from_single_energy(path: str) -> HeaderData:
    """Load data from an energy scan."""
    with fits.open(path) as hdul:
        header_data = HeaderData().read_file(hdul)
    return header_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "C:/Users/hduva/Washington State University (email.wsu.edu)/Carbon Lab Research Group - Documents/Synchrotron Logistics and Data/ALS - Berkeley/Data/BL1101/2023Nov/XRR/Processed/ZnPc/CCD Scan 82261/286.7/190.0/ZnPc82261-00348.fits"
    header_data = from_single_energy(path)
    print(header_data)

    cv.imshow("Image", header_data.image.processed)
    cv.waitKey(0)
