"""Fitting Utilities for XRR Fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from refnx._lib import (
    MapWrapper,
    emcee,
    flatten,
    possibly_open_file,
)
from refnx._lib import (
    unique as f_unique,
)
from refnx._lib.emcee.moves.de import DEMove
from refnx._lib.emcee.moves.gaussian import GaussianMove
from refnx._lib.util import getargspec
from refnx.analysis import CurveFitter, Objective, is_parameter, process_chain
from scipy._lib._util import check_random_state

if TYPE_CHECKING:
    from pyref.fitting import ReflectModel, XrayReflectDataset

demove = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7, gamma0=1), 0.05)]
gmove = GaussianMove(1e-7)


class AnisotropyObjective(Objective):
    """Objective for including an extra weight for anisotropy data."""

    def __init__(
        self,
        model: ReflectModel,
        data: XrayReflectDataset,
        logp_extra=None,
        ll_scale: float | None = 1.0,
        **kwargs,
    ):
        super().__init__(model, data, logp_extra=logp_extra, **kwargs)
        self.ll_scale = ll_scale

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
        model_anisotropy = self.model.anisotropy(self.data.anisotropy.x)
        data_anisotropy = self.data.anisotropy.y
        ll += 0.5 * np.sum((model_anisotropy - data_anisotropy) ** 2) * self.ll_scale
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
            self.data.anisotropy.x,
            self.model.anisotropy(self.data.anisotropy.x),
            color="C3",
            label="model",
        )
        ax_anisotropy.plot(
            self.data.anisotropy.x, self.data.anisotropy.y, color="C2", label="data"
        )

        ax.set_ylabel("Reflectivity")
        ax.legend()
        ax_anisotropy.legend()
        ax_anisotropy.axhline(0, color="k", ls="-", lw=plt.rcParams["axes.linewidth"])
        ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")

        return ax, ax_anisotropy


class Fitter(CurveFitter):
    """Overload the CurveFitter class to include custom sampling."""

    def __init__(
        self,
        objective: Any,
        nwalkers: int | None = None,
        ntemps: int = -1,
        **mcmc_kws: dict[Any],
    ) -> None:
        nparams = len(objective.varying_parameters())
        if nwalkers is None:
            nwalkers = max(2 * nparams, 200)
        elif nwalkers < 2 * nparams:
            import warnings

            nwalkers = 2 * nparams
            warnings.warn(
                f"Number of walkers should be at least 2 * nparams. "
                f"Setting nwalkers = {nwalkers}",
                stacklevel=2,
            )
        super().__init__(objective, nwalkers, ntemps, **mcmc_kws)

    def sample(
        self,
        steps,
        nthin=1,
        random_state=None,
        f=None,
        callback=None,
        verbose=True,
        pool=-1,
        **sampler_kws,
    ):
        """
        Sample from the objective.

        Parameters
        ----------
        steps : int
            Collect `steps` samples into the chain. The sampler will run a
            total of `steps * nthin` moves.
        nthin : int, optional
            Each chain sample is separated by `nthin` iterations.
        random_state : {None, int, `np.random.RandomState`, `np.random.Generator`}
            If performing MCMC with `ntemps == -1`:

            - If `random_state` is not specified the `~np.random.RandomState`
              singleton is used.
            - If `random_state` is an int, a new ``RandomState`` instance is
              used, seeded with `random_state`.
            - If `random_state` is already a ``RandomState`` instance, then
              that object is used.

            If using parallel tempering then random number generation is
            controlled by ``np.random.default_rng(random_state)``

            Specify `random_state` for repeatable minimizations.
        f : file-like or str
            File to incrementally save chain progress to. Each row in the file
            is a flattened array of size `(nwalkers, ndim)` or
            `(ntemps, nwalkers, ndim)`. There are `steps` rows in the
            file.
        callback : callable
            callback function to be called at each iteration step. Has the
            signature `callback(coords, logprob)`.
        verbose : bool, optional
            Gives updates on the sampling progress
        pool : int or map-like object, optional
            If `pool` is an `int` then it specifies the number of threads to
            use for parallelization. If `pool == -1`, then all CPU's are used.
            If pool is a map-like callable that follows the same calling
            sequence as the built-in map function, then this pool is used for
            parallelisation.

        sampler_kws : dict
            Keywords to pass to the sampler.sample method. Please see the corresponding
            method :meth:`emcee.EnsembleSampler.sample` or
            :meth:`ptemcee.sampler.Sampler.sample` for more information.

        Notes
        -----
        Please see :class:`emcee.EnsembleSampler` for its detailed behaviour.

        >>> # we'll burn the first 500 steps
        >>> fitter.sample(500)
        >>> # after you've run those, then discard them by resetting the
        >>> # sampler.
        >>> fitter.sampler.reset()
        >>> # Now collect 40 steps, each step separated by 50 sampler
        >>> # generations.
        >>> fitter.sample(40, nthin=50)

        One can also burn and thin in `Curvefitter.process_chain`.
        """
        self._check_vars_unchanged()

        # setup a random number generator
        # want Generator for ptemcee
        if self._ntemps == -1:
            rng = check_random_state(random_state)
            # require rng to be a RandomState
            if isinstance(random_state, np.random.Generator):
                rng = np.random.RandomState()
        else:
            rng = np.random.default_rng(random_state)

        if self._state is None:
            self.initialise(random_state=rng)

        # for saving progress to file
        def _callback_wrapper(state, h=None):
            if callback is not None:
                callback(state.coords, state.log_prob)

            if h is not None:
                h.write(" ".join(map(str, state.coords.ravel())))
                h.write("\n")

        # remove chains from each of the parameters because they slow down
        # pickling but only if they are parameter objects.
        flat_params = f_unique(flatten(self.objective.parameters))
        flat_params = [param for param in flat_params if is_parameter(param)]
        # zero out all the old parameter stderrs
        for param in flat_params:
            param.stderr = None
            param.chain = None

        # make sure the checkpoint file exists
        if f is not None:
            with possibly_open_file(f, "w") as h:
                # write the shape of each step of the chain
                h.write("# ")
                shape = self._state.coords.shape
                h.write(", ".join(map(str, shape)))
                h.write("\n")

        # set the random state of the sampler
        # normally one could give this as an argument to the sample method
        # but PTSampler didn't historically accept that...
        if self._ntemps == -1 and isinstance(rng, np.random.RandomState):
            rstate0 = rng.get_state()
            self._state.random_state = rstate0
            self.sampler.random_state = rstate0
        elif self._ntemps > 0:
            self._state.random_state = rng.bit_generator.state

        # Passthough sampler_kws to the sampler.sample method outside of the
        # parallelisation context.
        sampler_kws = {} if sampler_kws is None else sampler_kws
        sampler_args = getargspec(self.sampler.sample).args

        # update sampler_kws with the sampler_args from instantiated Fitter.
        if "progress" in sampler_args and verbose:
            sampler_kws["progress"] = True
            verbose = False
        if "thin_by" in sampler_kws:
            sampler_kws["thin_by"] = nthin
            sampler_kws.pop("thin", 0)

        sampler_kws.update({"iterations": steps, "thin": nthin})

        # using context manager means we kill off zombie pool objects
        # but does mean that the pool has to be specified each time.
        with MapWrapper(pool) as g, possibly_open_file(f, "a") as h:
            # if you're not creating more than 1 thread, then don't bother with
            # a pool.
            if isinstance(self.sampler, emcee.EnsembleSampler):
                if pool == 1:
                    self.sampler.pool = None
                else:
                    self.sampler.pool = g
            else:
                sampler_kws["mapper"] = g

            # perform the sampling
            for state in self.sampler.sample(self._state, **sampler_kws):
                self._state = state
                _callback_wrapper(state, h=h)

        if isinstance(self.sampler, emcee.EnsembleSampler):
            self.sampler.pool = None

        # sets parameter value and stderr
        return process_chain(self.objective, self.chain)


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
