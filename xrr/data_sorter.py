from pathlib import Path
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
try:
    from xrr.load_fits import MultiReader
    from xrr._config import FLAGS, HEADER_DICT
    from xrr.toolkit import FileDialog
except:
    from load_fits import MultiReader
    from _config import FLAGS, HEADER_DICT
    from toolkit import FileDialog


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


def main():
    dataPath = FileDialog.getDirectory(title="Select the Data Directory")
    sortedPath = FileDialog.getDirectory(title="Select the Sorted Directory")
    sampleName = input("What Is the Sample Name?")
    sortedPath = sortedPath / sampleName
    sortedStructure = "/-en/-pol"
    FitsSorter.sortDirectory(dataPath, sortedPath, sortedStructure)


if __name__ == "__main__":
    main()
