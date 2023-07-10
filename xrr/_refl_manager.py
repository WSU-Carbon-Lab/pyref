from cv2 import sqrBoxFilter, sqrt
from matplotlib import patches
from _image_manager import ImageProcs
from _load_fits import MultiReader, HEADER_VALUES
from _toolkit import XrayDomainTransform
from concurrent.futures import ThreadPoolExecutor
from typing import Final
import numpy as np
import pandas as pd


# COLUMN NAME STATIC VALUES
REFL_COLUMN_NAMES: Final[list] = ["Direct Beam Intensity", "Background Intensity"]
REFL_NAME: Final[str] = "Refl"
REFL_ERR_NAME: Final[str] = "Refl Err"
SCAT_NAME: Final[str] = "Q"


class ReflectivityProcs:
    """
    Using the _load_fits.MultiReader we can generate a dataframe containing the header data and a list containing images.
    Each ImageProcs method is vectorized allowing us to apply the procs to lists and arrays
    """

    @staticmethod
    def getBeamSpots(imageList: list, mask: np.ndarray | None = None):
        with ThreadPoolExecutor() as executor:
            trimmedImages = list(
                executor.map(lambda image: ImageProcs.removeEdge(image), imageList)
            )
            if mask != None:
                maskedImages = list(
                    executor.map(
                        lambda image: ImageProcs.applyMask(image, mask), trimmedImages
                    )
                )
                filteredImages = list(
                    executor.map(
                        lambda image: ImageProcs.medianFilter(image), maskedImages
                    )
                )
            else:
                maskedImages = trimmedImages
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
        return filteredImages, beamSpots, maskedImages

    @staticmethod
    def getSubImages(
        maskedImages: list[np.ndarray],
        beamSpots: list[tuple],
        height: int = 20,
        width: int = 20,
    ):
        with ThreadPoolExecutor() as executor:
            brightROI, darkROI = list(
                executor.map(
                    lambda args: ImageProcs.roiReduction(
                        args[0].shape, args[1], height=height, width=width
                    ),
                    zip(maskedImages, beamSpots),
                )
            )
        return brightROI, darkROI

    @staticmethod
    def computeRefl(reflBeamSpots: list, darkBeamSpots: list) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            refl_intensity = list(
                executor.map(lambda image: ImageProcs.sumImage(image), reflBeamSpots)
            )
            dark_intensity = list(
                executor.map(lambda image: ImageProcs.sumImage(image), darkBeamSpots)
            )
        df = pd.DataFrame(
            {REFL_COLUMN_NAMES[0]: refl_intensity, REFL_COLUMN_NAMES[1]: dark_intensity}
        )
        return df

    @staticmethod
    def buildDataFrame(metaData: pd.DataFrame, reflData: pd.DataFrame) -> pd.DataFrame:
        _metaDataErr(metaData)
        _reflDataErr(reflData)

        reflData.reset_index(drop=True, inplace=True)
        reflData[REFL_NAME] = (
            reflData[REFL_COLUMN_NAMES[0]] - reflData[REFL_COLUMN_NAMES[1]]
        )
        # / (metaData["Beam Current"] * metaData["Higher Order Suppressor"])

        refl_err = reflData[REFL_NAME] * np.sqrt(
            (1 / reflData[REFL_COLUMN_NAMES[0]]) - (1 / reflData[REFL_COLUMN_NAMES[1]])
        )

        scattering = XrayDomainTransform.toQ(
            metaData["Beamline Energy"], metaData["Sample Theta"]
        )
        data_frame = pd.DataFrame(
            {
                SCAT_NAME: scattering,
                REFL_NAME: reflData[REFL_NAME],
                REFL_ERR_NAME: refl_err,
            }
        )

        return data_frame

    @staticmethod
    def normalizeReflData(reflDataFrame: pd.DataFrame) -> pd.DataFrame:
        _izero_count = 0
        while True:
            if reflDataFrame[SCAT_NAME][_izero_count] == 0:
                _izero_count += 1
            else:
                break
        if _izero_count > 0:
            izero = np.average(reflDataFrame[REFL_NAME].iloc[: _izero_count - 1])
            izero_err_avg = np.average(
                reflDataFrame[REFL_ERR_NAME].iloc[: _izero_count - 1]
            )
            izero_err = izero_err_avg / sqrt(_izero_count)
        else:
            izero, izero_err = (1, 1)

        reflDataFrame[REFL_NAME] = reflDataFrame[REFL_NAME] / izero  # type: ignore
        reflDataFrame[REFL_ERR_NAME] = reflDataFrame[REFL_NAME] * np.sqrt(
            (izero_err / izero) ** 2
            + (reflDataFrame[REFL_ERR_NAME] / reflDataFrame[REFL_NAME]) ** 2
        )

        reflDataFrame.drop(reflDataFrame.index[:_izero_count])
        return reflDataFrame


def _metaDataErr(metaData: pd.DataFrame):
    global HEADER_VALUES
    assert np.all(metaData.columns == HEADER_VALUES)


def _reflDataErr(reflData: pd.DataFrame):
    global REFL_COLUMN_NAMES
    assert np.all(reflData.columns == REFL_COLUMN_NAMES)


def add_rectangle_to_image(image, rect_slice):
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Create a rectangle patch with dashed edge
    rect = patches.Rectangle(
        (rect_slice.start, rect_slice.stop),
        rect_slice.step,
        rect_slice.start - rect_slice.stop,
        linewidth=1,
        edgecolor="white",
        linestyle="dashed",
        facecolor="none",
    )


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.patches as patches

    meta, images = MultiReader.readFile(
        Path("tests/TestData/Sorted/ZnPc_P100_E180276/282.5/190.0").resolve()
    )
    filtered_image, beam_spots, masked_image = ReflectivityProcs.getBeamSpots(images)
    x, y = ReflectivityProcs.getSubImages(masked_image, beam_spots)
