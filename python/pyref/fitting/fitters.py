"""Fitting Utilities for XRR Fitting."""

from __future__ import annotations

import pickle as pkl
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from refnx._lib.emcee.moves.de import DEMove
from refnx._lib.emcee.moves.gaussian import GaussianMove
from refnx.analysis import CurveFitter

if TYPE_CHECKING:
    from refnx.analysis import GlobalObjective, Objective

demove = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7, gamma0=1), 0.05)]
gmove = GaussianMove(1e-7)


@dataclass
class Fitter:
    """Fitter class for reflectometry data."""

    obj: Objective | GlobalObjective
    en: str
    walkers_per_param: int = 10
    burn_in: float = 0.1

    def __post_init__(self):
        """Initialise the fitter object."""
        self.move = [
            (DEMove(sigma=1e-7), 0.90),
            (DEMove(sigma=1e-7, gamma0=1), 0.1),
        ]
        self._n_walkers = self.walkers_per_param * len(self.obj.varying_parameters())
        self._init_fitter()

    def _init_fitter(self):
        self.fitter = CurveFitter(self.obj, nwalkers=self._n_walkers, moves=self.move)

    @property
    def n_params(self):
        """Number of parameters in the model."""
        return len(self.obj.data.data[0]) - len(self.obj.varying_parameters())

    @cached_property
    def red_chisqr(self):
        """Reduced chi-squared value."""
        return self.obj.chisqr() / self.n_params

    @cached_property
    def log_likelihood(self):
        """Calculate the log-likelihood."""
        chisqr = self.obj.chisqr()
        # Assuming Gaussian errors, the log-likelihood is proportional to -0.5 * chisqr
        return -0.5 * chisqr

    def fit(
        self,
        steps_per_param: int = 10,
        thin: int = 1,
        seed: int = 1,
        init: Literal["jitter", "prior"] = "jitter",
        *,
        show_output: bool = False,
    ):
        """Fit the reflectometry data."""
        steps = steps_per_param * self.n_params

        self.fitter.initialise(init, random_state=seed)
        self.chain = self.fitter.sample(
            steps,
            random_state=seed,
            nthin=thin,
        )

        print(f"Reduced χ2 = {self.red_chisqr}")
        print(f"Log-likelihood = {self.log_likelihood}")  # Add log-likelihood output

        if self.red_chisqr > 1.5:
            self.move = [
                (DEMove(sigma=1e-7), 0.90),
            ]
            self._init_fitter()
            self.chain = self.fitter.sample(
                steps,
                random_state=seed,
                nthin=thin,
            )
        else:
            self.move = [
                (GaussianMove(1e-7)),
            ]
            self._init_fitter()
            self.chain = self.fitter.sample(
                steps,
                random_state=seed,
                nthin=thin,
            )

        if show_output:
            self.show_output()

    def show_output(self):
        """Display fitting results and save to file."""
        print(rf"Reduced χ2 = {self.red_chisqr}")
        print(self.obj.varying_parameters())

        # Plot log posterior
        fig, ax = plt.subplots()
        ax.plot(-self.fitter.logpost)
        fig.show()

        # Plot residuals and model structure
        self.obj.plot()
        plt.show()
        self.obj.model.structure.plot()

    def export(self, filename: str | None = None):
        """Export the fitter object to a pickle file."""
        if filename is None:
            filename = f"{self.en}.pkl"
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pkl.dump(self, f)


class LogpExtra:
    """Log Prior Constraint for the fitting of reflectometry data."""

    def __init__(self, objective):
        self.objective = objective

    def __call__(self, model, data):
        """Apply custom log-prior constraint."""
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(), "thick")
            rough_pars = sort_pars(pars.flattened(), "rough")
            for i in range(len(rough_pars)):
                if rough_pars[i].vary or thick_pars[i].vary:
                    interface_limit = np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                    if float(thick_pars[i].value - interface_limit) < 0:
                        return -np.inf
        return 0


def sort_pars(pars, str_check, vary=None, str_not=" "):
    """Sort parameters based on the string in the name."""
    return [
        par
        for par in pars
        if str_check in par.name
        and str_not not in par.name
        and (vary is None or par.vary == vary)
    ]
