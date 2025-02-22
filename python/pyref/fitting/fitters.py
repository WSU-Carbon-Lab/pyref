"""Fitting Utilities for XRR Fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from refnx._lib.emcee.moves.de import DEMove
from refnx._lib.emcee.moves.gaussian import GaussianMove
from refnx.analysis import Objective

if TYPE_CHECKING:
    from pyref.fitting import ReflectDataset, XrayReflectDataset

demove = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7, gamma0=1), 0.05)]
gmove = GaussianMove(1e-7)


class AnisotropyObjective(Objective):
    """Objective for including an extra weight for anisotropy data."""

    def __init__(
        self, model: XrayReflectDataset, data: ReflectDataset, logp_extra=None, **kwargs
    ):
        super().__init__(model, data, logp_extra=logp_extra, **kwargs)

    # ----------/ Custom Log-Posterior /----------
    def logl(self, pvals=None):
        """
        Calculate the log-likelhood of the system.

        The major component of the log-likelhood probability is from the data.
        Extra potential terms are added on from the Model, `self.model.logp`,
        and the user specifiable `logp_extra` function.

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        logl : float
            log-likelihood probability

        Notes
        -----
        The log-likelihood is calculated as:

        .. code-block:: python

            logl = -0.5 * np.sum(((y - model) / s_n) ** 2 + np.log(2 * pi * s_n**2))
            logp += self.model.logp()
            logp += self.logp_extra(self.model, self.data)

        where

        .. code-block:: python

            s_n**2 = y_err**2 + exp(2 * lnsigma) * model**2

        """
        ll = super().logl(pvals=pvals)
        model_anisotropy = self.model.anisotropy(self.data.x)[1]
        data_anisotropy = self.data.anisotropy.y
        ll += -np.sum((model_anisotropy - data_anisotropy) ** 2)
        return ll

    # ----------/ Custom Plotting /----------
    def plot(
        self,
        samples=0,
        pvals=None,
        data=None,
        model=None,
        ax=None,
        ax_anisotropy=None,
        **kwargs,
    ):
        """Plot function that includes anisotropy information."""
        if ax is None:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(
                nrows=2,
                sharex=False,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax = axs[0]
            ax_anisotropy = axs[1]

        y, y_err, model = self._data_transform(model=self.generative())
        if self.weighted:
            ax.errorbar(
                self.data.x,
                y,
                y_err,
                label=self.data.name,
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=2,
                capsize=2,
            )
        else:
            ax.plot(self.data.x, y, label=self.data.name)

        if samples > 0:
            # Get a number of chains, chosen randomly, set the objective,
            # and plot the model.
            models = []
            for curve in self._generate_generative_mcmc(ngen=samples):
                _, _, model = self._data_transform(model=curve)
                models.append(model)
            models = np.array(models)
            # find the max and min of the models and fill between them
            ax.fill_between(
                self.data.x,
                np.percentile(models, 16, axis=0),
                np.percentile(models, 84, axis=0),
                color="C1",
                alpha=0.5,
            )
            ax.fill_between(
                self.data.x,
                np.percentile(models, 2.5, axis=0),
                np.percentile(models, 97.5, axis=0),
                color="C1",
                alpha=0.2,
            )
        # add the fit
        ax.plot(self.data.x, model, color="C1", label="fit", zorder=20, **kwargs)

        ax_anisotropy.set_ylabel("Anisotropy")

        ax_anisotropy.plot(
            *self.model.anisotropy(self.data.x), color="C3", label="model"
        )
        ax_anisotropy.plot(
            self.data.anisotropy.x, self.data.anisotropy.y, color="C2", label="data"
        )

        ax.set_ylabel("Reflectivity")
        ax.legend()
        ax_anisotropy.legend()
        ax_anisotropy.axhline(0, color="k", ls="--", lw=0.5)
        ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")

        return ax, ax_anisotropy


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
