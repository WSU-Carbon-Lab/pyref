"""Main module."""
import pandas as pd
import numpy as np
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


class XRR:
    def __init__(
        self,
        dataDir: None | str | Path = None,
        saveDir: None | str | Path = None,
        reuse: bool = False,
        fresh: bool = True,
        *reflargs,
        **reflkwargs
    ):
        self.directory = dataDir
        self._mask = None
        if reuse:
            (
                self.refl,
                self.images,
                self.masked,
                self.filtered,
                self.beam,
                self.backrgound,
            ) = Reuse.openForReuse(self.directory)

        else:
            meta, self.images, self.directory = MultiReader()(directory=self.directory, fresh=fresh)  # type: ignore
            (
                self.refl,
                self.masked,
                self.filtered,
                self.beam,
                self.backrgound,
            ) = ReflProcs()(meta, self.images, self._mask, *reflargs, **reflkwargs)
            if saveDir == None:
                saveDir = f"{str(self.directory.parent)}_{self.directory.name}"
        Reuse.saveForReuse(
            saveDir,
            self.refl,
            self.images,
            self.masked,
            self.filtered,
            self.beam,
            self.backrgound,
        )

    def __call__(self, mask=None, *reflargs, **reflkwargs) -> Any:
        self._mask = mask
        self.refl, self.masked, self.filtered, self.beam, self.backrgound = ReflProcs()(
            self.refl, self.images, self._mask, *reflargs, **reflkwargs
        )

    def __str__(self):
        return self.refl.__str__()

    def plot(self, *args, **kwargs):
        self.refl.plot(
            x=Q_VEC_NAME,
            y=REFL_NAME,
            yerr=REFL_ERR_NAME,
            kind="scatter",
            logy=True,
            *args,
            **kwargs
        )


class MultiEnergyXRR:
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


    @staticmethod
    def getXRR(path: Path, fresh: bool):
        energy = path.parent
        xrr = XRR(dataDir=path,fresh=fresh)
        return energy.name, xrr


if __name__ == "__main__":
    # test = XRR(fresh=True)
    # print(test)
    # # test.plot()
    # # plt.show()

    test2 = MultiEnergyXRR()
    print(test2)
