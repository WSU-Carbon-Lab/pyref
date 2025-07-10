"""Fitting Utilities for XRR Fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import sigfig
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
from refnx._lib.util import getargspec
from refnx.analysis import (
    CurveFitter,
    Objective,
    is_parameter,
    process_chain,
)
from scipy._lib._util import check_random_state

from pyref.fitting.io import XrayReflectDataset
from pyref.fitting.model import ReflectModel

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd
    import xarray as xr
    from refnx.analysis import GlobalObjective, Interval, Parameter

    from pyref.fitting import ReflectModel

demove = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7, gamma0=1), 0.05)]
MA = np.asin(np.sqrt(2 / 3))


class AnisotropyObjective(Objective):
    """Objective for including an extra weight for anisotropy data."""

    def __init__(
        self,
        model: ReflectModel,
        data: XrayReflectDataset,
        logp_extra=None,
        logp_anisotropy_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(model, data, logp_extra=logp_extra, **kwargs)
        self.logp_anisotropy_weight = logp_anisotropy_weight

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
        if isinstance(self.data, XrayReflectDataset):
            ll *= 1 - self.logp_anisotropy_weight
            model_anisotropy = self.model.anisotropy(self.data.anisotropy.x)
            data_anisotropy = self.data.anisotropy.y
            ll += (
                -0.5
                * np.sum((model_anisotropy - data_anisotropy) ** 2)
                * self.logp_anisotropy_weight
            )
        ll /= len(self.data.x)
        return ll

    # ----------/ Custom Plotting /----------
    def plot(  # type: ignore
        self,
        samples=0,
        model=None,
        ax=None,
        ax_anisotropy=None,
        data_kwargs=None,
        model_kwargs=None,
        show_s=True,
        show_p=True,
        show_anisotropy=True,
    ):
        """
        Plot function that includes anisotropy information.

        Parameters
        ----------
        samples : int, optional
            Number of sample curves to plot from MCMC chain
        model : array-like, optional
            Model data to plot
        ax : matplotlib.Axes, optional
            Axes for reflectivity plot
        ax_anisotropy : matplotlib.Axes, optional
            Axes for anisotropy plot
        data_kwargs : dict, optional
            Keyword arguments for data plotting
        model_kwargs : dict, optional
            Keyword arguments for model plotting
        show_s : bool, optional
            Whether to show s-polarization data
        show_p : bool, optional
            Whether to show p-polarization data
        show_anisotropy : bool, optional
            Whether to show anisotropy plot

        Returns
        -------
        tuple
            (ax, ax_anisotropy) - matplotlib axes objects
        """
        import matplotlib.pyplot as plt

        if data_kwargs is None:
            data_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        # Set up axes
        if ax is None:
            if show_anisotropy:
                fig, axs = plt.subplots(
                    nrows=2,
                    sharex=False,
                    figsize=(8, 6),
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                ax = axs[0]
                ax_anisotropy = axs[1]
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax_anisotropy = None
        elif ax_anisotropy is None and show_anisotropy:
            # Get the figure from the provided axis
            fig = ax.figure
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
            ax_anisotropy = fig.add_subplot(gs[1], sharex=ax)

        # Check if we have separate s and p polarization data
        has_separate_pol = hasattr(self.data, "s") and hasattr(self.data, "p")

        # Plot data
        if has_separate_pol:
            # Plot s-polarization if requested
            if show_s:
                ax.errorbar(
                    self.data.s.x,  # type: ignore
                    self.data.s.y,  # type: ignore
                    self.data.s.y_err,  # type: ignore
                    label=f"{self.data.name} s-pol" if self.data.name else "s-pol",
                    marker="o",
                    color="C0",
                    ms=3,
                    lw=0,
                    elinewidth=1,
                    capsize=1,
                    ecolor="k",
                    **data_kwargs,
                )

                # Calculate s-polarization model
                original_pol = self.model.pol
                self.model.pol = "s"
                s_model = self.model(self.data.s.x)  # type: ignore
                ax.plot(
                    self.data.s.x,  # type: ignore
                    s_model,
                    color="C0",
                    label="s-pol fit",
                    zorder=20,
                    **model_kwargs,
                )
                self.model.pol = original_pol

            # Plot p-polarization if requested
            if show_p:
                ax.errorbar(
                    self.data.p.x,  # type: ignore
                    self.data.p.y,  # type: ignore
                    self.data.p.y_err,  # type: ignore
                    label=f"{self.data.name} p-pol" if self.data.name else "p-pol",
                    marker="o",
                    color="C1",
                    ms=3,
                    lw=0,
                    elinewidth=1,
                    capsize=1,
                    ecolor="k",
                    **data_kwargs,
                )

                # Calculate p-polarization model
                original_pol = self.model.pol
                self.model.pol = "p"
                p_model = self.model(self.data.p.x)  # type: ignore
                ax.plot(
                    self.data.p.x,  # type: ignore
                    p_model,
                    color="C1",
                    label="p-pol fit",
                    zorder=20,
                    **model_kwargs,
                )
                self.model.pol = original_pol
        else:
            # Handle combined data case
            ax.errorbar(
                self.data.x,
                self.data.y,
                self.data.y_err,
                label=self.data.name,
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
                **data_kwargs,
            )

            # Plot combined model
            model = self.generative()
            _, _, model_transformed = self._data_transform(model=model)

            if samples > 0:
                # Get sample curves from MCMC chain
                models = []
                for curve in self._generate_generative_mcmc(ngen=samples):
                    _, _, model_t = self._data_transform(model=curve)
                    models.append(model_t)
                models = np.array(models)

                # Show 1-sigma and 2-sigma confidence intervals
                ax.fill_between(
                    self.data.x,
                    np.percentile(models, 16, axis=0),  # type: ignore
                    np.percentile(models, 84, axis=0),  # type: ignore
                    color="C1",
                    alpha=0.5,
                )
                ax.fill_between(
                    self.data.x,
                    np.percentile(models, 2.5, axis=0),  # type: ignore
                    np.percentile(models, 97.5, axis=0),  # type: ignore
                    color="C1",
                    alpha=0.2,
                )

            # Plot the fit
            ax.plot(
                self.data.x,
                model_transformed,  # type: ignore
                color="C1",
                label="fit",
                zorder=20,
                **model_kwargs,
            )

        # Plot anisotropy if enabled
        if (
            ax_anisotropy is not None
            and show_anisotropy
            and hasattr(self.data, "anisotropy")
        ):
            ax_anisotropy.set_ylabel("Anisotropy")

            # Plot anisotropy model
            ax_anisotropy.plot(
                self.data.anisotropy.x,  # type: ignore
                self.model.anisotropy(self.data.anisotropy.x),  # type: ignore
                color="C3",
                zorder=20,
                label="model",
            )

            # Plot anisotropy data
            ax_anisotropy.plot(
                self.data.anisotropy.x,  # type: ignore
                self.data.anisotropy.y,  # type: ignore
                color="C3",
                marker="o",
                markersize=3,
                linestyle="None",
                label="data",
            )

            ax_anisotropy.legend()
            ax_anisotropy.axhline(
                0, color="k", ls="-", lw=plt.rcParams["axes.linewidth"]
            )
            ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")

        # Finalize styling
        ax.set_ylabel("Reflectivity")
        ax.set_yscale("log")
        ax.legend()

        return ax, ax_anisotropy

    @classmethod
    def build_objective(
        cls, model: ReflectModel, data: XrayReflectDataset, ani_weight: float = 0.5
    ) -> AnisotropyObjective:
        """
        Build a new AnisotropyObjective.

        Parameters
        ----------
        model : ReflectModel
            The reflectivity model to use.
        data : XrayReflectDataset
            The dataset containing the reflectivity data.
        ani_weight : float, optional
            Weight for the anisotropy data in the log-likelihood calculation.

        Returns
        -------
        AnisotropyObjective
            A new AnisotropyObjective instance.
        """
        obj: AnisotropyObjective = cls(
            model=model, data=data, logp_anisotropy_weight=ani_weight
        )
        lpe: LogpExtra = LogpExtra(obj)
        obj.logp_extra = lpe
        return obj


class Fitter(CurveFitter):
    """Overload the CurveFitter class to include custom sampling."""

    def __init__(
        self,
        objective: Objective | AnisotropyObjective | GlobalObjective,
        ntemps: int = -1,
        nwalkers: int | None = None,
        walkers_per_param: int = 10,
        **mcmc_kws: Any,
    ) -> None:
        nparams = len(objective.varying_parameters())
        ideal_walkers = nparams * walkers_per_param
        if nwalkers is None:
            nwalkers = max(ideal_walkers, 200)
        elif nwalkers < ideal_walkers:
            import warnings

            nwalkers = ideal_walkers
            warnings.warn(
                f"Number of walkers should be at least {ideal_walkers}. "
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
        skip_check=False,
    ):
        """
        Perform sampling from the objective.

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

        # set the random state of the sampler
        # normally one could give this as an argument to the sample method
        # but PTSampler didn't historically accept that...
        if self._ntemps == -1 and isinstance(rng, np.random.RandomState):
            rstate0 = rng.get_state()
            self._state.random_state = rstate0
            self.sampler.random_state = rstate0
        elif self._ntemps > 0:
            self._state.random_state = rng.bit_generator.state  # type: ignore

        # Passthough sampler_kws to the sampler.sample method outside of the
        # parallelisation context.
        sampler_kws = {}
        sampler_args = getargspec(self.sampler.sample).args

        # update sampler_kws with the sampler_args from instantiated Fitter.
        if "progress" in sampler_args and verbose:
            sampler_kws["progress"] = True
            verbose = False
        if "thin_by" in sampler_kws:
            sampler_kws["thin_by"] = nthin
            sampler_kws.pop("thin", 0)

        sampler_kws.update({"iterations": steps, "thin": nthin})
        sampler_kws.update({"skip_initial_state_check": skip_check})

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
                self._state = state  # type: ignore
                _callback_wrapper(state, h=h)

        if isinstance(self.sampler, emcee.EnsembleSampler):
            self.sampler.pool = None

        # sets parameter value and stderr
        return process_chain(self.objective, self.chain)

    @property
    def chain(self) -> np.ndarray:
        """Get the chain from the sampler."""
        chain = self.sampler.get_chain()
        if chain is None:
            msg = "Sampler chain is not available."
            raise ValueError(msg)
        return chain

    def to_arviz(self) -> az.InferenceData:
        """
        Convert MCMC results to an ArviZ InferenceData object.

        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard.
        thin : int, optional
            Thinning factor for the chain.

        Returns
        -------
        arviz.InferenceData
            ArviZ InferenceData object.
        """
        var_names = [p.name for p in self.objective.varying_parameters()]
        return az.from_emcee(
            sampler=self.sampler,
            var_names=var_names,
            coords={"chain": np.arange(self._nwalkers)},
        )

    def diagnose(self, **kwargs) -> pd.DataFrame | xr.Dataset:
        """
        Run ArviZ diagnostics on the MCMC chain.

        Returns
        -------
        DataFrame or Dataset
            ArviZ diagnostics results.
        """
        return az.summary(self.to_arviz(), **kwargs)


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


def rounded_values(
    parameter: Parameter,
) -> tuple[float, float]:
    """Round the values and errors to n significant figures."""
    x = float(parameter.value)
    xerr: Literal[0] | float = parameter.stderr if parameter.stderr else 0
    if xerr > 0:
        err = float(sigfig.round(xerr, 1))  # type: ignore
        val = round(x, -int(np.floor(np.log10(err))))  # type: ignore
    else:
        err = xerr
        val = x
    return val, err


def _fix_bound(parameter: Parameter, nsigma=5, *, by_bounds=False) -> None:
    val, err = rounded_values(parameter)
    bounds: Interval = parameter.bounds
    # round err and value to proper sig figs
    if not by_bounds and bounds is not None:
        min_val: float = float(sigfig.round(val - nsigma * err, 1))  # type: ignore
        max_val: float = float(sigfig.round(val + nsigma * err, 1))  # type: ignore
    else:
        spread = bounds.ub - bounds.lb
        min_val = val - spread / 2
        max_val = val + spread / 2
    if "thick" in parameter.name or "rough" in parameter.name:  # type: ignore
        min_val = 0
    if parameter.name.split("_")[-1] == "rho":  # type: ignore
        min_val = 1
        max_val += 0.5
    if "rotation" in parameter.name:  # type: ignore
        max_val = np.radians(90)
        min_val = MA
    if err > 0:
        better_bounds = (min_val, max_val)
        parameter.setp(bounds=better_bounds)


def _fix_bounds(obj, nsigma=5, by_bounds=False) -> None:
    params = obj.varying_parameters()
    for p in params:
        _fix_bound(p, nsigma=nsigma, by_bounds=by_bounds)


def fit(
    obj, recursion_limit=2, workers=-1, **kwargs
) -> tuple[AnisotropyObjective, Fitter]:
    """
    Fit the model to the data using the provided objective.

    Parameters
    ----------
    obj : AnisotropyObjective
        The objective function to minimize.

    Returns
    -------
    tuple[AnisotropyObjective, Fitter]
        The fitted objective and the fitter used.
    """
    import copy

    objective: AnisotropyObjective = copy.deepcopy(obj)
    fitter: Fitter = Fitter(objective, **kwargs)
    target = "nlpost"  # nlpost accounts for uncertainty in the data
    # Recursively use the differential evolution algorithm to locate the minima
    for _ in range(recursion_limit):
        fitter.fit(
            "differential_evolution",
            target=target,
            workers=workers,
        )
        _fix_bounds(objective, by_bounds=True)
    # Once you are in the minima based on the recursive DE, use L-BFGS-B to refine
    fitter.fit("L-BFGS-B", target=target, options={"workers": workers})
    _fix_bounds(objective, by_bounds=False)  # user error estimates to zoom parameters
    # Finally, use MCMC to sample the posterior distribution
    fitter.sample(
        steps=1000,
        nthin=10,
        random_state=42,
    )
    return objective, fitter
