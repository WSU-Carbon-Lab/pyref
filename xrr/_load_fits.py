from pathlib import Path
from typing import Final
from astropy.io import fits
import numpy as np
import pandas as pd


HEADER_VALUES: Final[list] = [
    "Beamline Energy",
    "Sample Theta",
    "Beam Current",
    "Higher Order Suppressor",
    "EPU Polarization",
]


class FitsReader:
    @staticmethod
    def readHeader(fitsFilePath: Path, headerValues: list = HEADER_VALUES) -> dict:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore

        return {
            key: round(headerData[key], 4) for key in headerValues if key in headerData
        }

    @staticmethod
    def readImage(fitsFilePath: Path) -> list:
        with fits.open(fitsFilePath) as hdul:
            imageData = hdul[2].data  # type: ignore

        return imageData

    @staticmethod
    def readFile(
        fitsFilePath: Path, headerValues: list = HEADER_VALUES
    ) -> tuple[pd.DataFrame, list]:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore
            imageData = hdul[2].data  # type: ignore

        headerDict = {
            key: round(headerData[key], 4) for key in headerValues if key in headerData
        }
        return pd.DataFrame(headerDict, index=[0]), imageData


class MultiReader:
    @staticmethod
    def readHeader(
        dataFilePath: Path, headerValues: list = HEADER_VALUES, fileName: bool = False
    ) -> pd.DataFrame:
        headerDFList = []
        for file in dataFilePath.glob("*.fits"):
            headerDF = pd.DataFrame(
                FitsReader.readHeader(file, headerValues=headerValues), index=[0]
            )
            if fileName:
                headerDF["File Path"] = file
            headerDFList.append(headerDF)

        return pd.concat(headerDFList)

    @staticmethod
    def readImage(dataFilePath: Path) -> list:
        imageList = []
        for file in dataFilePath.glob("*.fits"):
            imageList.append(FitsReader.readImage(file))

        return imageList

    @staticmethod
    def readFile(
        dataFilePath: Path, headerValues: list = HEADER_VALUES
    ) -> tuple[pd.DataFrame, list]:
        imageList = []
        headerDFList = []
        for file in dataFilePath.glob("*.glob"):
            headerDFList.append(
                pd.DataFrame(
                    FitsReader.readHeader(file, headerValues=headerValues), index=[0]
                )
            )
            imageList.append(FitsReader.readImage(file))

        return pd.concat(headerDFList), imageList


def _construct_tests():
    multi_fits_directory = Path("tests/TestData/Sorted/282.5/")
    fits_directory = Path("tests/TestData/TestFits.fits")

    single_df, single_u = FitsReader.readFile(fits_directory)
    single_df.to_csv("tests/TestData/TestSingleDataFrame.csv", index=False)  # type: ignore
    np.savetxt("tests/TestData/TestSingleImage.txt", single_u)  # type: ignore

    df, u = MultiReader.readFile(multi_fits_directory)
    df.to_csv("tests/TestData/TestDataFrame.csv", index=False)


if __name__ == "__main__":
    ###########################################################################
    # # Use this to construct new test cases if needed
    _construct_tests()

    ###########################################################################
    # The outcome of this test should run approximately 12 sec

    # import timeit

    # fits_directory = Path("tests/TestData/TestFits.fits")
    # # Measure the execution time for the original loadMultipleFits function
    # timer = timeit.Timer(lambda: ExperimentReader.readFile(fits_directory))

    # # Print the execution times
    # print("Original Load Time:", timer.timeit(100))
