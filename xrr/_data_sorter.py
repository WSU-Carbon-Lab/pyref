from pathlib import Path
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
from _load_fits import MultiReader


FLAGS = {
    "-en": "Beamline Energy",
    "-pol": "EPU Polarization",
    "-n": "File Path",
}


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
            append = str(round(row[flag], 1))

        targetDirectory = targetDirectory / append
        targetDirectory.mkdir(exist_ok=True)

    copy2(sourceFile, targetDirectory)


def _getSampleName(string: str) -> str:
    """General File Structure "Sample-ScanID.fits" this parses sample"""
    fileName = Path(string).name.split(".")
    return fileName[0].split("-")[0]


if __name__ == "__main__":
    test_path = Path("tests/TestData/CCD").resolve()
    test_sorted = Path("tests/TestData/Sorted").resolve()
    test_structure = "/-n/-en/-pol"
    FitsSorter.sortDirectory(test_path, test_sorted, test_structure)
