"""Main module."""
import warnings
import pandas as pd
import numpy as np
from abc import ABC, abstractclassmethod
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Final
from refl_manager import (
    ReflProcs,
    REFL_COLUMN_NAMES,
    REFL_NAME,
    REFL_ERR_NAME,
    Q_VEC_NAME,
)
from load_fits import MultiReader
from refl_reuse import Reuse
from toolkit import FileDialog

POL_NAMES: Final[list] = ["100.0", "190.0"]

class DataBackend(ABC):
    @abstractclassmethod
    def getData(self):
        pass

class Reuseable(ABC):
    @abstractclassmethod
    def saveData(self):
        pass

    @abstractclassmethod
    def openData(self):
        pass

class Maskable(ABC):

    @abstractclassmethod
    def setMask(self):
        pass

class DataFrontEnd(ABC):
    def plot(self):
        pass

    def display(self):
        pass

class ProcessedRefl(DataBackend, Reuseable, Maskable):
    def getData(self, obj, directory: Path | None = None, mask: np.ndarray | None = None):
        if directory == None:
            obj.dataPath = FileDialog.getDirectory()        
        else:
            obj.dataPath = directory
        obj.mask = mask
        metadata, obj.images = MultiReader.readFile(obj.dataPath)
        obj.refl, obj.maskedImages, obj.filteredImages, obj.beamSpots, obj.darkSpots = ReflProcs.main(metadata, obj.images, mask)
    
    def saveData(self, obj, saveDir: Path):
        Reuse.saveForReuse(saveDir, obj.refl, obj.images, obj.maskedImages, obj.filteredImages, obj.beamSpots, obj.darkSpots) 
        
    def openData(self, obj, openDir: Path | None):
        if openDir == None:
            openDir = FileDialog.getDirectory()        
        else:
            openDir = openDir

        obj.refl, obj.images, obj.maskedImages, obj.filteredImages, obj.beamSpots, obj.darkSpots = Reuse.openForReuse(openDir)
    
    def setMask(self, obj, mask):
        obj.mask = mask
        self.getData(obj, directory = obj.dataPath, mask = mask)



class DebuggingRefl(DataBackend):
    ...


class XRR(DataFrontEnd):
    def __init__(self, directory: Path | None = None, fresh: bool = True, mask: np.ndarray|None = None):
        self.dataSource = ProcessedRefl()
        self.mask = mask
        if fresh:
            self.dataSource.getData(self, directory, mask = self.mask)
        else:
            self.dataSource.openData(self, directory)

        self.savePath = self.dataPath.parent # type: ignore
        self.dataSource.saveData(self, self.savePath)
    
    def saveData(self):
        self.dataSource.saveData(self, self.savePath)
    
    def setMask(self, mask):
        self.dataSource.setMask(self, mask)
    

    def plot(self, *args, **kwargs):
        self.refl.plot(x=Q_VEC_NAME, y=REFL_NAME, yerr=REFL_ERR_NAME, kind='scatter', *args, **kwargs) # type: ignore
    



class MultiEnergyXRR(DataFrontEnd):
    def __init__(self, fresh: bool = True) -> None:
        self.sampleDirectory: Path = FileDialog.getDirectory()  # type: ignore
        self.P100_Energies: list[Path] = [
            energy / POL_NAMES[0]
            for energy in self.sampleDirectory.iterdir()
            if energy.is_dir() and (energy / POL_NAMES[0]).exists()
        ]
        if len(self.P100_Energies) > 0:
            self.Pol100 = {
                en: xrr
                for energyDir in self.P100_Energies
                for en, xrr in [MultiEnergyXRR.getXRR(energyDir, fresh)]
            }

        self.P190_Energies: list[Path] = [
            energy / POL_NAMES[1]
            for energy in self.sampleDirectory.iterdir()
            if energy.is_dir() and (energy / POL_NAMES[1]).exists()
        ]
        if len(self.P190_Energies) > 0:
            self.Pol190 = {
                en: xrr
                for energyDir in self.P190_Energies
                for en, xrr in [MultiEnergyXRR.getXRR(energyDir, fresh)]
            }



def checkExist(obj, parameters):
    for param in parameters:
        if not hasattr(obj, param):
            return False
    return True

if __name__ == "__main__":
    # test = XRR(fresh=True)
    # print(test)
    # # test.plot()
    # # plt.show()

    test2 = MultiEnergyXRR()
    print(test2)
