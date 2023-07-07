from _image_manager import ImageProcs
from _load_fits import MultiReader
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd


class ReflectivityProcs:
    """
    Using the _load_fits.MultiReader we can generate a dataframe containing the header data and a list containing images.
    Each ImageProcs method is vectorized allowing us to apply the procs to lists and arrays
    """

    @staticmethod
    def selectBrightSpots(imageList: list, height: int = 10, width: int = 10):
        with ThreadPoolExecutor() as executor:
            trimmedImages = list(
                executor.map(lambda image: ImageProcs.removeEdge(image), imageList)
            )
            filteredImages = list(
                executor.map(
                    lambda image: ImageProcs.medianFilter(image), trimmedImages
                )
            )
            beamSpots = list(
                executor.map(
                    lambda image: ImageProcs.findMaximum(image), filteredImages
                )
            )

        return beamSpots


if __name__ == "__main__":
    from pathlib import Path

    images = MultiReader.readImage(
        Path("tests/TestData/Sorted/ZnPc_P100_E180276/282.5/190.0").resolve()
    )
    spots = ReflectivityProcs.selectBrightSpots(images)
    print(spots)
