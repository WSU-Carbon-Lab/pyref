"""Main module."""
from abc import ABC, abstractclassmethod
from typing import Literal, Final
from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

try:
    from xrr.refl_manager import ReflProcs
    from xrr.load_fits import MultiReader
    from xrr.refl_reuse import Reuse
    from xrr._config import REFL_COLUMN_NAMES
except:
    from refl_manager import ReflProcs
    from load_fits import MultiReader
    from refl_reuse import Reuse
    from _config import REFL_COLUMN_NAMES


class Refl:
    """
    This is the main reflectivity front end interface. The class is initialized using the backend.getData method. This initialized several class properties,
    ----------------------------------------------------------------------------
    #########################      Properties          #########################
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

    images: list
        This is a list of numpy arrays. Each numpy array is the raw image data from the fits file

    masked: list
        This is a list of numpy arrays. Each numpy array is the image data with a mask applied.

    filtered: list
        This is a list of numpy arrays. Each numpy array is the masked image with a median filter applied.

    beamspot: list
        This is a list of numpy arrays. Each numpy array is the beamspot location on the raw data set.

    background: list
        This is a list of numpy arrays. Each numpy array is located on the opposite side of the image from the beamspot location.
    //// Note: These parameters are initialized as booleans as they take up a single bite of data. This is simply present for typesetting purposes.
    ----------------------------------------------------------------------------
    #########################         Methods          #########################
    ----------------------------------------------------------------------------
    mask: np.ndarray
        Property with setter and getter methods. This sets and gets masked attribute from the backend and re-initializes the object.

    saveData, plot, display, debug
        Inherited from the backend
    """

    def __init__(
        self,
        path: Path | None = None,
        backendKey: Literal["single", "multi"] = "single",
        *backArgs,
        **backKWArgs
    ):
        global BACKEND

        self.path = path
        self.refl: pd.DataFrame = True  # type: ignore
        self.images: list = True  # type: ignore
        self.masked: list = True  # type: ignore
        self.filtered: list = True  # type: ignore
        self.beamspot: list = True  # type: ignore
        self.background: list = True  # type: ignore

        self.backendKey: Literal["single", "multi"] = backendKey
        self.backendProcessor = BACKEND[backendKey](*backArgs, **backKWArgs)
        self.backendProcessor.getData(self, path)
        self.backendProcessor.saveData(self)

    @property
    def mask(self):
        if not self.mask is None:
            plt.imshow(self.mask)
        else:
            return self.mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        backKWArgs = {"mask": mask}
        self.__init__(path=self.path, backendKey=self.backendKey, **backKWArgs)

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
        metadata, obj.images, obj.path = MultiReader.main(obj.path, **dataKWArgs)
        ReflProcs.main(obj, mask, metadata, source=source)

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

    def display(self, obj: Refl):
        fig = px.scatter(
            obj.refl,
            x=REFL_COLUMN_NAMES["Q"],
            y=REFL_COLUMN_NAMES["R"],
            error_y=REFL_COLUMN_NAMES["R Err"],
            log_y=True,
            hover_data=list(REFL_COLUMN_NAMES.values()),
        )
        fig.show()

    def debug(self, obj: Refl):
        ...


class MultiRefl(DataBackend):
    def getData(self, obj: Refl):
        ...

    def saveData(self, obj: Refl):
        ...

    def plot(self, obj: Refl):
        ...

    def display(self, obj: Refl):
        ...

    def debug(self, obj: Refl):
        ...


BACKEND: Final[dict] = {
    "single": SingleRefl,
    "multi": MultiRefl,
}

if __name__ == "__main__":
    test = Refl()
    test.display()
