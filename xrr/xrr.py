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

POL_NAMES: Final[list] = ["P100", "P190"]


class XRR:
    def __init__(
        self,
        directory: None | str | Path = None,
        fresh: bool = True,
        dialog: bool = True,
        *reflargs,
        **reflkwargs
    ):
        self._mask = None
        if not fresh:
            self.refl, self.images, self.masked, self.filtered, self.beam, self.backrgound = Reuse.openForReuse(directory)
        else:
            meta, self.images, self.directory = MultiReader()(directory=directory, dialog=dialog)  # type: ignore
            self.refl, self.masked, self.filtered, self.beam, self.backrgound = ReflProcs()(
                meta, self.images, self._mask, *reflargs, **reflkwargs
            )
        Reuse.saveForReuse(
            self.directory,
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
        ax = self.refl.plot(
            x=Q_VEC_NAME,
            y=REFL_NAME,
            yerr=REFL_ERR_NAME,
            kind="scatter",
            logy=True,
            *args,
            **kwargs
        )
        return ax


class MultiEnergyXRR:
    def __init__(self) -> None:
        SampleDirectory: Path = FileDialog.getDirectory()  # type: ignore
        P100_Energies: list[Path] = [
            energy / POL_NAMES[0]
            for energy in SampleDirectory.iterdir()
            if energy.is_dir()
        ]
        P190_Energies: list[Path] = [
            energy / POL_NAMES[1]
            for energy in SampleDirectory.iterdir()
            if energy.is_dir()
        ]


if __name__ == "__main__":
    test = XRR(fresh=True)
    print(test)
    test.plot()
    plt.show()
