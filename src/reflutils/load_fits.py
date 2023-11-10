import pandas as pd
from pathlib import Path
from astropy.io import fits
from shutil import copy2
import click
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from reflutils._config import HEADER_LIST, HEADER_DICT, FLAGS
    from reflutils.toolkit import FileDialog
except:
    from _config import HEADER_LIST, HEADER_DICT, FLAGS
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

                assert len(stitchDF) == len(imageList)
                headerDFList = []
                imageList = []

            headerDFList.append(headerDF)
            imageList.append(FitsReader.readImage(file))

        return headerStitchedDFList, imageStitchedList


class FitsSorter:
    @staticmethod
    def sortDirectory(dataPath: Path, sortedPath: Path, sortedStructure: str) -> None:
        flags = FitsSorter._getFlags(
            sortedPath=sortedPath, sortedStructure=sortedStructure
        )
        FitsSorter._sortByFlags(dataPath, sortedPath, flags)

    @staticmethod
    def _getFlags(sortedPath: Path, sortedStructure: str) -> list:
        global FLAGS

        structureChildren = sortedStructure.split("/")

        directories = []
        flags = []
        for child in structureChildren:
            if child.startswith("-"):
                flags.append(FLAGS[child])
            else:
                directories.append(child)
            directoryPath = sortedPath / "/".join(directories)
            directoryPath.mkdir(exist_ok=True)
        return flags

    @staticmethod
    def _sortByFlags(dataPath: Path, sortedPath: Path, flags: list):
        headerDF = MultiReader.readHeader(dataPath, headerValues=flags, fileName=True)
        headerDF.reindex(columns=flags)
        with ThreadPoolExecutor() as executor:
            list(
                executor.map(
                    lambda row: copyFile(row[1], sortedPath=sortedPath, flags=flags),
                    headerDF.iterrows(),
                )
            )


def copyFile(row, sortedPath: Path, flags: list) -> None:
    sourceFile: Path = row[-1]  # type: ignore
    targetDirectory = sortedPath

    for flag in flags:
        if flag == FLAGS["-n"]:
            append = _getSampleName(row[flag])
        else:
            append = str(round(row[HEADER_DICT[flag]], 1))

        targetDirectory = targetDirectory / append
        targetDirectory.mkdir(exist_ok=True)

    copy2(sourceFile, targetDirectory)


def _getSampleName(string: str) -> str:
    """General File Structure "Sample-ScanID.fits" this parses sample"""
    fileName = Path(string).name.split(".")
    return fileName[0].split("-")[0]


class IoStream:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.sample_name = self.sampleName()
        self.scan_id = self.scanId()

    def sampleName(self):
        sample_list = []
        for file in self.data_path.glob("*.fits"):
            s = file.name.split(".")
            sample = s.rstrip("123456789")
            if sample not in sample_list:
                sample_list.append(sample)
        self.sample_name = sample_list

    def scanId(self):
        scan_list = []
        for file in self.data_path.glob("*.fits"):
            s = file.name.split(".")
            scan = s.lstrip(self.sample_name)
            run = scan.split("-")[0]
            if scan not in scan_list:
                scan_list.append(scan)
        self.scan_id = scan_list


@click.group()
@click.argument("dir", default=os.getcwd())
def cli():
    reader = MultiReader()


@click.command()
def init():
    dataPath = FileDialog.getDirectory(title="Select the Data Directory")
    sortedPath = FileDialog.getDirectory(title="Select the Sorted Directory")


@click.command()
def main():
    dataPath = FileDialog.getDirectory(title="Select the Data Directory")
    sortedPath = FileDialog.getDirectory(title="Select the Sorted Directory")
    sampleName = input("What Is the Sample Name?")
    sortedPath = sortedPath / sampleName
    sortedStructure = "/-en/-pol"
    FitsSorter.sortDirectory(dataPath, sortedPath, sortedStructure)


if __name__ == "__main__":
    main()
