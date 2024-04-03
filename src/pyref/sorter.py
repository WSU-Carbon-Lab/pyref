from pathlib import Path
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor

from .load_fits import MultiReader
from ._config import FLAGS, HEADER_DICT
from .toolkit import FileDialog


def main():
    dataPath = FileDialog.getDirectory(title="Select the Data Directory")
    sortedPath = FileDialog.getDirectory(title="Select the Sorted Directory")
    sampleName = input("What Is the Sample Name?")
    sortedPath = sortedPath / sampleName
    sortedStructure = "/-en/-pol"
    FitsSorter.sortDirectory(dataPath, sortedPath, sortedStructure)


if __name__ == "__main__":
    main()
