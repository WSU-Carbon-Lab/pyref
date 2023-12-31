from email.mime import image
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

try:
    from reflutils._config import HEADER_LIST, HEADER_DICT
    from reflutils.toolkit import FileDialog
except:
    from _config import HEADER_LIST, HEADER_DICT
    from toolkit import FileDialog


class FitsReader:
    @staticmethod
    def readHeader(fitsFilePath: Path, headerValues: list[str] = HEADER_LIST) -> dict:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore

        return {
            HEADER_DICT[key]: round(headerData[key], 4)
            for key in headerValues
            if key in headerData
        }

    @staticmethod
    def readImage(fitsFilePath: Path) -> list:
        with fits.open(fitsFilePath) as hdul:
            imageData = hdul[2].data  # type: ignore

        return imageData

    @staticmethod
    def readFile(
        fitsFilePath: Path, headerValues: list[str] = HEADER_LIST
    ) -> tuple[pd.DataFrame, list]:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore
            imageData = hdul[2].data  # type: ignore

        headerDict = {
            HEADER_DICT[key]: round(headerData[key], 4)
            for key in headerValues
            if key in headerData
        }
        return pd.DataFrame(headerDict, index=[0]), imageData


class MultiReader:
    @staticmethod
    def readHeader(
        dataFilePath: Path,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
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
        dataFilePath: Path | None,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
    ) -> tuple[pd.DataFrame, list]:
        if dataFilePath == None:
            raise ValueError("Restart operation and choose a file path")
        imageList = []
        headerDFList = []
        for i, file in enumerate(dataFilePath.glob("*.fits")):
            headerDF = pd.DataFrame(
                FitsReader.readHeader(file, headerValues=headerValues), index=[i]
            )
            if fileName:
                headerDF["File Path"] = file
            headerDFList.append(headerDF)
            imageList.append(FitsReader.readImage(file))

        return pd.concat(headerDFList), imageList

    @staticmethod
    def prepareReflData(
        dataFilePath: Path | None,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
    ) -> tuple[list[pd.DataFrame], list[list]]:
        if dataFilePath == None:
            raise ValueError("Restart operation and choose a file path")

        imageList = []
        headerDFList = []
        headerStitchedDFList = []
        imageStitchedList = []
        offset = 0
        for i, file in enumerate(dataFilePath.glob("*.fits")):
            headerDF = pd.DataFrame(
                FitsReader.readHeader(file, headerValues=headerValues),
                index=[i],
            )
            if fileName:
                headerDF["File Path"] = file
                
            if (
                i > 0
                and headerDF[HEADER_DICT["Sample Theta"]].iat[0]
                < headerDFList[-1][HEADER_DICT["Sample Theta"]].iat[0]
            ) or (i == len(list(dataFilePath.glob("*.fits"))) - 1):
                offset += i + 1
                stitchDF = pd.concat(headerDFList).reset_index(drop=True)
                headerStitchedDFList.append(stitchDF)
                imageStitchedList.append(imageList)

                assert(len(stitchDF) == len(imageList))
                headerDFList = []
                imageList = []

            headerDFList.append(headerDF)
            imageList.append(FitsReader.readImage(file))

        return headerStitchedDFList, imageStitchedList


def _constructTests():
    multi_fits_directory = Path(
        "tests/TestData/Sorted/ZnPc_P100_E180276/282.5/190.0"
    ).resolve()
    fits_directory = Path(
        "tests/TestData/Sorted/ZnPc_P100_E180276/282.5/190.0/ZnPc_P100_E180276-00001.fits"
    ).resolve()
    test_single_df = Path("tests/TestData/TestSingleDataFrame.csv").resolve()
    test_single_u = Path("tests/TestData/TestSingleImage.txt").resolve()
    test_df = Path("tests/TestData/TestDataFrame.csv").resolve()

    single_df, single_u = FitsReader.readFile(fits_directory)
    single_df.to_csv(test_single_df, index=False)  # type: ignore
    np.savetxt(test_single_u, single_u)  # type: ignore

    df, _ = MultiReader.readFile(multi_fits_directory)
    df.to_csv(test_df, index=False)


if __name__ == "__main__":
    ###########################################################################
    # # Use this to construct new test cases if needed
    _constructTests()

    ###########################################################################
    # The outcome of this test should run approximately 12 sec

    # import timeit

    # fits_directory = Path("tests/TestData/CCD").resolve()
    # # Measure the execution time for the original loadMultipleFits function
    # timer1 = timeit.Timer(lambda: MultiReader.readHeader(fits_directory))

    # # Print the execution times
    # print("Original Load Time:", timer1.timeit(10))
