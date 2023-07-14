"""Main module."""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from refl_manager import ReflProcs
from refl_manager import REFL_COLUMN_NAMES, REFL_NAME, REFL_ERR_NAME, Q_VEC_NAME
from load_fits import MultiReader


class XRR:
    def __init__(self, fresh: bool = True, *reflargs, **reflkwargs):
        self._mask = None
        meta, self.images = MultiReader()(fresh=fresh)
        self.refl, self.masked, self.filtered, self.beam, self.backrgound = ReflProcs()(
            meta, self.images, self._mask, *reflargs, **reflkwargs
        )

    def __call__(self, mask=None, *reflargs, **reflkwargs) -> Any:
        self._mask = mask
        self.refl, self.masked, self.filtered, self.beam, self.backrgound = ReflProcs()(
            self.refl, self.images, self._mask, *reflargs, **reflkwargs
        )
    
    def __str__(self):
        return self.refl.__str__()

    def plot(self, *args, **kwargs):
        ax = self.refl.plot(x=Q_VEC_NAME, y = REFL_NAME, yerr = REFL_ERR_NAME, kind = 'scatter', logy=True, * args, **kwargs)
        return ax


if __name__ == "__main__":
    test = XRR(fresh=True)
    print(test)
    test.plot()
    plt.show()