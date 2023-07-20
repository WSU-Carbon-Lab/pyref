"""Main module."""
from abc import ABC, abstractclassmethod
from typing import Literal, Final
from pathlib import Path
from warnings import warn
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

try:
    from xrr.refl_manager import ReflProcs
    from xrr.load_fits import MultiReader
    from xrr.refl_reuse import Reuse
    from xrr.toolkit import FileDialog
    from xrr._config import REFL_COLUMN_NAMES, POL
except:
    from refl_manager import ReflProcs
    from load_fits import MultiReader
    from refl_reuse import Reuse
    from toolkit import FileDialog
    from _config import REFL_COLUMN_NAMES, POL


class Refl:
    """
    Main Reflectivity Interface
    ----------------------------------------------------------------------------
    ## Properties
    ----------------------------------------------------------------------------
    refl: DataFrame
        This is a DataFrame with the following columns
        >>> Beamline Energy
        >>> Sample Theta
        >>> Beam Current
        >>> Higher Order Suppressor
        >>> EPU Polarization
        >>> Direct Beam Intensity
        >>> Background Intensity
        >>> Refl
        >>> Refl Err
        >>> Q

    images: DataFrame
        This DataFrame of image ArrayLike objects with the following columns
        >>> Images
        >>> Masked
        >>> Filtered
        >>> Beam Image
        >>> Dark Image
        
    mask: ArrayLike [bool]
        This boolean array is used to mask the data. Using Set Mask
    
    path: Path
        pathlib object pointing to the location of the dataset
    
    energies: float | list[float]
        depending on the backend, gives the energies represented in the Refl object

    polarizations: float | list[float]
        depending on the backend, gives the polarization represented in the Refl object

    At it's base, this is really just a wrapper for a pandas DataFrame, but for more advanced backends this structure becomes more rich to contain all the needed data.
    ----------------------------------------------------------------------------
    ## Methods         
    ----------------------------------------------------------------------------
    mask: np.ndarray
        Property with setter and getter methods. This sets and gets masked attribute from the backend and re-initializes the object.

    saveData, plot, display, debug
        Inherited from the backend
    """

    def __init__(
        self,
        path: Path | None = None,
        backend: Literal["single", "multi"] = "single",
        *backArgs,
        **backKWArgs
    ):
        global BACKEND

        self.path = path
        self.refl: pd.DataFrame = True  #type: ignore
        self.images: pd.DataFrame = True #type: ignore
        self.energies: list[str] | str = True #type: ignore
        self.polarization: list[tuple[str, str]] | str = True #type: ignore

        self.backendKey: Literal["single", "multi"] = backend
        self.backendProcessor = BACKEND[backend](*backArgs, **backKWArgs)
        self.backendProcessor.getData(self)
        self.backendProcessor.saveData(self)

    def __str__(self) -> str:
        return self.refl.__str__()

    @property
    def mask(self):
        if not self.mask is None:
            plt.imshow(self.mask)
        else:
            return self.mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        backKWArgs = {"mask": mask}
        self.__init__(path=self.path, backend=self.backendKey, **backKWArgs)

    def saveData(self, savePath):
        self.backendProcessor.saveData(self, savePath)

    def plot(self, *pltArgs, **pltKWArgs):
        self.backendProcessor.plot(self, *pltArgs, **pltKWArgs)

    def display(self, *dispArgs, **dispKWArgs):
        self.backendProcessor.display(self, *dispArgs, **dispKWArgs)

    def debug(self, *dispArgs, **dispKWArgs):
        self.backendProcessor.debug(self, *dispArgs, **dispKWArgs)


class DataBackend(ABC):
    """
    Abstract class method for data backends. Each backend has the following methods
    ----------------------------------------------------------------------------

    getData:
        This method will take take in the front end class and add attributes to
        it. This will be called in the front end constructor. As an outline,

        >>> getData(self, obj, from):
        >>>     if from == 'fits':
        >>>         getFits()
        >>>         ...
        >>>     elif from == 'saved':
        >>>         getSaved()


    saveData:
        This method will save the backend data in a specified format. The Refl DataFrame will be saved as a csv with no index. Every image dataset will instead be saved as a .npz file. As an outline,

        >>> saveData(self, obj, dataPath):
        >>>     savePath = str(dataPath.parent)
        >>>     obj.refl.to_csv(savePath + '.csv', index = False)
        >>>     np.savez(savePath + '.npz')

    plot:
        This method will plot the xrr data. The xrr data needs to be plotted as Q vs Refl with the Refl axis as a log scale.

    display:
        This method is the general workforce display method. Ideally, this should use a HoloViz to display data point information and the CCD images that are used to construct the reflectivity point.

    debug:
        This method uses the display method, but displays different information than the classic display method. In particular,

        >>> Beam Intensity vs Q                                  (Raw Intensity)
        >>> Dark Intensity vs Q                               (Background Noise)
        >>> Beam Intensity / Dark Intensity vs Q         (Signal To Noise Ratio)
        >>> Refl vs Q                                              (Final Plots)
    """

    @abstractclassmethod
    def getData(self):
        pass

    @abstractclassmethod
    def saveData(self):
        pass

    @abstractclassmethod
    def plot(self):
        pass

    @abstractclassmethod
    def display(self):
        pass

    @abstractclassmethod
    def debug(self):
        pass


class SingleRefl(DataBackend):
    def getData(
        self,
        obj: Refl,
        mask: np.ndarray | None = None,
        source: Literal["fits", "csv"] = "fits",
        **dataKWArgs
    ):
        if isinstance(obj.path, type(None)):
            obj.path = FileDialog.getDirectory(title='Choose Single Polarization Directory')

        obj.energies = obj.path.name
        obj.polarization = obj.path.name
        
        metadata, images = MultiReader.readFile(
            obj.path, **dataKWArgs
        )

        if source == 'fits':
            obj.images, beamSpots, darkSpots = ReflProcs.getBeamSpots(
                images, mask=mask
            )
            obj.images = ReflProcs.getSubImages(
                obj.images, beamSpots, darkSpots
            )
            pureReflDF = ReflProcs.getDf(metadata, obj.images)
            obj.refl = ReflProcs.scaleSeries(pureReflDF, **dataKWArgs)

        elif source == 'csv':
            Reuse.openForReuse(obj)

        else:
            raise ValueError("Invalid Data Source - choose 'csv' or 'fits'")

    def saveData(self, obj: Refl):
        Reuse.saveForReuse(obj)

    def plot(self, obj: Refl, *args, **kwargs):
        obj.refl.plot(
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["R"],
            yerr=REFL_COLUMN_NAMES["R Err"],
            logy=True,
            kind="scatter",
            *args,
            **kwargs
        )
        plt.show()

    def display(self, obj: Refl):
        fig = px.scatter(
            obj.refl,
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["R"],
            error_y=REFL_COLUMN_NAMES["R Err"],
            log_y=True,
            hover_data=list(['Energy', 'Theta', 'Current', 'HOS', 'POL', 'Intensity', 'Background', 'RawRefl']),
        )
        fig.show()

    def debug(self, obj: Refl):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        obj.refl.plot(
            ax=axes[0, 0],
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["Beam Spot"],
            logy=True,
            kind="scatter",
        )
        obj.refl.plot(
            ax=axes[0, 1],
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["Dark Spot"],
            logy=True,
            kind="scatter",
        )
        obj.refl.plot(
            ax=axes[1, 0],
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["Raw"],
            logy=True,
            kind="scatter",
        )
        obj.refl.plot(
            ax=axes[1, 1],
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["R"],
            logy=True,
            kind="scatter",
        )
        plt.show()

class MultiRefl(DataBackend):
    def getData(
        self,
        obj: Refl,
        mask: np.ndarray | None = None,
        source: Literal["fits", "csv"] = "fits",
        **dataKWArgs
    ):
        if isinstance(obj.path, type(None)):
            obj.path = FileDialog.getDirectory(title="Choose Single Sample Directory")

        obj.energies = [en.name for en in obj.path.iterdir() if en.is_dir()]
        obj.polarization = []

        if source == 'fits':
            EN_reflList = []
            EN_imageList = []
            for energy in obj.energies:
                energyDir = obj.path / energy
                if not energyDir.exists():
                    raise ValueError(f"Invalid data directory - path structure should be sample/{energy}. Invalid path: {str(energyDir)}")
                pols = [pol.name for pol in energyDir.iterdir() if pol.is_dir()]
                POL_reflList = []
                POL_imageList = []
                for pol in pols:
                    dataDir = energyDir / pol
                    if not energyDir.exists():
                        warn(f"No experimental data found for polarization: {pol} at {energy}")

                    if dataDir.exists():
                        metadata, images = MultiReader.readFile(
                            dataDir, **dataKWArgs
                        )

                        imageList, beamSpots, darkSpots = ReflProcs.           getBeamSpots(
                                    images, mask=mask
                        )
                        images = ReflProcs.getSubImages(
                            imageList, beamSpots, darkSpots
                        )
                        pureReflDF = ReflProcs.getDf(metadata, images)
                        refl = ReflProcs.scaleSeries(pureReflDF, **dataKWArgs)
                        POL_reflList.append(refl)
                        POL_imageList.append(images)
                EN_reflList.append(pd.concat(POL_reflList, axis=1, keys=pols))
                EN_imageList.append(pd.concat(POL_imageList, axis=1, keys=pols))
                obj.polarization.append(tuple(pols))

        elif source == 'csv':
            EN_reflList = []
            EN_imageList = []
            obj.polarization = []
            for energy in obj.energies:
                POL_reflList = []
                POL_imageList = []
                pols = [str(file.name).split('_')[1] for file in obj.path.glob(f'*{energy}_*')  if file.is_file()]         
                for pol in pols:
                    refl, images = Reuse.multiOpen(obj.path, energy, pol)
                    POL_reflList.append(refl)
                    POL_imageList.append(images)
                    POL_reflList.append(refl)
                    POL_imageList.append(images)
                EN_reflList.append(pd.concat(POL_reflList, axis=1, keys=pols))
                EN_imageList.append(pd.concat(POL_imageList, axis=1, keys=pols))
                obj.polarization.append(tuple(pols))
        else:
            raise ValueError("Invalid Data Source - choose 'csv' or 'fits'")

        refl = pd.concat(EN_reflList, axis = 1, keys=obj.energies)
        images = pd.concat(EN_imageList, axis = 1, keys=obj.energies)

        refl.index.name = 'Index'
        refl.columns.names = ['ENERGY', 'POL', 'REFL']
        images.index.name = 'Index'
        images.columns.names = ['ENERGY', 'POL', 'IMAGES']

        obj.refl = refl
        obj.images = images

    def saveData(self, obj: Refl):
        ...

    def plot(self, obj: Refl, kind: Literal['en', 'pol'], *args, **kwargs):
        if kind == 'en':
            fig, axes = plt.subplots(ncols=2)
            for i, pol in enumerate(obj.polarization[0]):
                axes[i].set_xlabel(REFL_COLUMN_NAMES['Q'])
                axes[i].set_ylabel(REFL_COLUMN_NAMES['R'])
                axes[i].set_title(f"{pol}")
                for j, en in enumerate(obj.energies):
                    scale = 10**(1.5*j)
                    x = obj.refl[en][pol][REFL_COLUMN_NAMES['Q']]
                    y = scale * obj.refl[en][pol][REFL_COLUMN_NAMES['R']]
                    yerr = scale * obj.refl[en][pol][REFL_COLUMN_NAMES['R Err']]
                    axes[i].errorbar(x,y, yerr = yerr, fmt = '.', label = f'{en}')
                    xmax = max(x)
                    if j == 0:
                        axes[i].set_ylim(bottom = min(y)/2)
                    axes[i].set_ylim(top = scale)
                    if axes[i].get_xlim()[1] > xmax:
                        axes[i].set_xlim(right = xmax + .001)
                    axes[i].set_xlim(left = 0)

                axes[i].set_yscale('log')
                axes[i].legend()
            plt.show()



        elif kind == 'pol':
            ...
        else:
            raise ValueError("Invalid plot kind - choose 'en' or 'pol'")

    def display(self, obj: Refl):
        ...

    def debug(self, obj: Refl):
        ...


BACKEND: Final[dict] = {
    "single": SingleRefl,
    "multi": MultiRefl,
}

if __name__ == "__main__":
    test1 = Refl(backend = 'multi')
    test1.plot(kind = 'en')
    print(test1)
