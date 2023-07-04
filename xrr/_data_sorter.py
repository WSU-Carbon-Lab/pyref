from pathlib import Path
from _load_fits import MultiReader
import os

FLAGS = {"-en": "Beamline Energy", "-pol": "EPU Polarization"}


class FitsSorter:
    @staticmethod
    def sortDirectory(dataPath: Path, sortedPath: Path, sortedStructure: str):
        flags = FitsSorter._getFlags(
            sortedPath=sortedPath, sortedStructure=sortedStructure
        )

    @staticmethod
    def _getFlags(sortedPath: Path, sortedStructure: str):
        structureChildren = sortedStructure.split("/")

        directories = []
        flags = []
        for child in structureChildren:
            if child.startswith("-"):
                flags.append(FLAGS[child])
            else:
                directories.append(child)
            directoryPath = sortedPath / "/".join(directories)
            os.mkdir(directoryPath)
        return flags

    @staticmethod
    def sortByHeader(dataPath: Path, flags: list):
        headerDF = MultiReader.readHeader(dataPath, headerValues=flags)
        for flag in flags:
            os.mkdir()
