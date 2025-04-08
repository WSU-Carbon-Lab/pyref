"""Fitting Utilities for XRR Fitting."""

from __future__ import annotations

import re
from pathlib import Path
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
from refnx.analysis import (
    CurveFitter,
    Objective,
    is_parameter,
    process_chain,
)
from scipy._lib._util import check_random_state

if TYPE_CHECKING:
    from refnx.analysis import (
        GlobalObjective,
    )

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
        # Normalize the anisotropy log-likelihood by the number of points
        ll /= len(data_anisotropy)
        # Add the log-prior constraint
        return ll

    # ----------/ Custom Plotting /----------
    def plot(
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
                    self.data.s.x,
                    self.data.s.y,
                    self.data.s.y_err,
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
                s_model = self.model(self.data.s.x)
                ax.plot(
                    self.data.s.x,
                    s_model,
                    color="C1",
                    label="s-pol fit",
                    zorder=20,
                    **model_kwargs,
                )
                self.model.pol = original_pol

            # Plot p-polarization if requested
            if show_p:
                ax.errorbar(
                    self.data.p.x,
                    self.data.p.y,
                    self.data.p.y_err,
                    label=f"{self.data.name} p-pol" if self.data.name else "p-pol",
                    marker="o",
                    color="C2",
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
                p_model = self.model(self.data.p.x)
                ax.plot(
                    self.data.p.x,
                    p_model,
                    color="C3",
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

            # Plot the fit
            ax.plot(
                self.data.x,
                model_transformed,
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
                self.data.anisotropy.x,
                self.model.anisotropy(self.data.anisotropy.x),
                color="C3",
                label="model",
            )

            # Plot anisotropy data
            ax_anisotropy.plot(
                self.data.anisotropy.x,
                self.data.anisotropy.y,
                color="C2",
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


class Fitter(CurveFitter):
    """Overload the CurveFitter class to include custom sampling."""

    def __init__(
        self,
        objective: Objective | AnisotropyObjective | GlobalObjective,
        ntemps: int = -1,
        nwalkers: int | None = None,
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

        # Store the backend for checkpointing
        self.backend = None

    def _get_checkpoint_filename(self, f=None):
        """Generate a default checkpoint filename based on the objective name."""
        if f is not None:
            return f

        if hasattr(self.objective, "data") and hasattr(self.objective.data, "name"):
            # Use the dataset name if available
            base_name = self.objective.data.name
            if not base_name:
                base_name = "mcmc_checkpoint"
        else:
            # Default name if no objective name is available
            base_name = "mcmc_checkpoint"

        # Clean the name to be a valid filename
        base_name = re.sub(r"[^\w\-_\.]", "_", base_name)

        return f"{base_name}.h5"

    def setup_backend(self, f=None, use_checkpoint=True):
        """
        Set up the HDF5 backend for the sampler.

        Parameters
        ----------
        f : str, optional
            Filename for the HDF5 backend. If None, a name will be generated based on \
            the objective.
        use_checkpoint : bool, optional
            Whether to try to resume from an existing checkpoint file.

        Returns
        -------
        backend : emcee.backends.HDFBackend
            The configured backend object
        """
        from emcee.backends import HDFBackend

        # Generate a default filename if none provided
        filename = self._get_checkpoint_filename(f)

        # Create the backend
        backend = HDFBackend(filename)

        # Check if we should use an existing checkpoint
        if use_checkpoint and Path.exists(filename) and backend.iteration > 0:
            print(f"Resuming from checkpoint file: {filename}")
            print(f"Previous iterations: {backend.iteration}")
            # Set the initial state from the last position in the chain
            self._state = emcee.State(
                coords=backend.get_chain()[-1],
                log_prob=backend.get_log_prob()[-1],
                random_state=None,  # This will be set in the sample method
            )
        else:
            # If not using checkpoint or no checkpoint exists, reset the backend
            print(f"Creating new checkpoint file: {filename}")
            backend.reset(self._nwalkers, len(self.objective.varying_parameters()))

        return backend

    def sample(
        self,
        steps,
        nthin=1,
        random_state=None,
        f=None,
        use_checkpoint=True,
        callback=None,
        verbose=True,
        pool=-1,
        **sampler_kws,
    ):
        """
        Sample from the objective using HDF5 backend for checkpointing.

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
        f : str, optional
            Filename for the HDF5 backend. If None, a name will be generated based on
            the objective.
        use_checkpoint : bool, optional
            Whether to try to resume from an existing checkpoint file.
        callback : callable, optional
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

        Returns
        -------
        res : np.ndarray
            The sampler chain
        """
        self._check_vars_unchanged()

        # Set up backend if we're using emcee (not parallel tempering)
        if self._ntemps == -1:
            # Set up the backend for checkpointing
            self.backend = self.setup_backend(f, use_checkpoint)

            # If we're using an existing backend, we need to recreate the sampler
            # with the backend attached
            if self.backend.iteration > 0 or self.sampler is None:
                self.sampler = emcee.EnsembleSampler(
                    self._nwalkers,
                    len(self.objective.varying_parameters()),
                    self.objective.logpost,
                    moves=demove,
                    backend=self.backend,
                )
        else:
            # For parallel tempering (ptemcee), use the traditional approach
            # since ptemcee doesn't support the HDF5 backend in the same way
            if use_checkpoint:
                print("Warning: HDF5 checkpoint not supported with parallel tempering.")
                print("Using traditional text-based checkpointing instead.")

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

        # for saving progress to file (only needed for ptemcee)
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

        # make sure the checkpoint file exists for ptemcee
        if f is not None and self._ntemps > 0:
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
        with MapWrapper(pool) as g:
            # For ptemcee we still need the file context
            if self._ntemps > 0:
                with possibly_open_file(f, "a") as h:
                    _run_sampler(self, g, sampler_kws, h, _callback_wrapper)
            else:
                _run_sampler(self, g, sampler_kws, None, callback)

        if isinstance(self.sampler, emcee.EnsembleSampler):
            self.sampler.pool = None

        # sets parameter value and stderr
        return process_chain(self.objective, self.chain)

    @property
    def chain(self):
        """Get the chain from the sampler, properly formatted."""
        if self._ntemps == -1:
            # For emcee with HDF5 backend
            if hasattr(self, "backend") and self.backend is not None:
                return self.backend.get_chain()
            if hasattr(self.sampler, "get_chain"):
                return self.sampler.get_chain()
            # Fallback for older emcee versions
            return self.sampler.chain
        else:
            # For ptemcee
            return self.sampler.chain

    def to_arviz(self, burn=0, thin=1):
        """
        Convert MCMC results to an ArviZ InferenceData object.

        Parameters
        ----------
        burn : int, optional
            Discard this many steps from the beginning of the chain
        thin : int, optional
            Keep every `thin` steps from the chain

        Returns
        -------
        arviz.InferenceData
            ArviZ InferenceData object containing posterior samples
        """
        try:
            import arviz as az
            import xarray as xr
        except ImportError as err:
            e = (
                "ArviZ and xarray are required for this feature. "
                "Install them with 'pip install arviz xarray'"
            )
            raise ImportError(e) from err

        # Get parameter names
        var_param_names = [p.name for p in self.objective.varying_parameters()]

        # Get chain, log_prob, and acceptance_fraction
        if self._ntemps == -1:  # emcee
            if hasattr(self, "backend") and self.backend is not None:
                chain = self.backend.get_chain()[burn::thin]
                log_prob = self.backend.get_log_prob()[burn::thin]
                try:
                    acceptance = self.backend.get_acceptance_fraction()
                except AttributeError:
                    acceptance = (
                        np.ones(self._nwalkers)
                        * self.sampler.acceptance_fraction.mean()
                    )
            else:
                chain = self.sampler.get_chain()[burn::thin]
                log_prob = self.sampler.get_log_prob()[burn::thin]
                acceptance = self.sampler.acceptance_fraction
        else:  # ptemcee
            chain = self.sampler.chain[0, :, burn::thin, :]
            log_prob = self.sampler.logprobability[0, :, burn::thin]
            acceptance = self.sampler.acceptance_fraction[0]

        # Convert to xarray Dataset for posterior
        n_steps, n_walkers, n_dims = chain.shape
        posterior_data = {
            var: (["draw", "chain", "param"], chain[:, :, i, np.newaxis])
            for i, var in enumerate(var_param_names)
        }

        # Add merged samples (all chains combined)
        for i, var in enumerate(var_param_names):
            flat_samples = chain[:, :, i].flatten()
            posterior_data[f"{var}_merged"] = (["sample"], flat_samples)

        # Create coords for the dataset
        coords = {
            "chain": np.arange(n_walkers),
            "draw": np.arange(n_steps),
            "param": [0],  # Just a dimension of 1 for parameters
            "sample": np.arange(n_steps * n_walkers),
        }

        # Create the xarray dataset
        posterior = xr.Dataset(posterior_data, coords=coords)

        # Create sample_stats dataset with log_probability
        sample_stats_data = {
            "log_likelihood": (["draw", "chain"], log_prob),
            "acceptance_rate": (["chain"], acceptance),
        }

        sample_stats = xr.Dataset(
            sample_stats_data,
            coords={
                "chain": np.arange(n_walkers),
                "draw": np.arange(n_steps),
            },
        )

        # Create InferenceData object
        return az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    def visualize(self, burn=0, thin=1, var_names=None, figsize=None):
        """
        Visualize MCMC results using ArviZ.

        Parameters
        ----------
        burn : int, optional
            Discard this many steps from the beginning of the chain
        thin : int, optional
            Keep every `thin` steps from the chain
        var_names : list of str, optional
            Variables to include in the plot (default: all)
        figsize : tuple, optional
            Figure size for the plot

        Returns
        -------
        arviz.plots.PairPlot
            ArviZ pair plot object
        """
        try:
            import arviz as az
        except ImportError as err:
            e = (
                "ArviZ and xarray are required for this feature. "
                "Install them with 'pip install arviz xarray'"
            )
            raise ImportError(e) from err

        # Convert to ArviZ InferenceData
        data = self.to_arviz(burn=burn, thin=thin)

        # If var_names not specified, use all varying parameters
        if var_names is None:
            var_names = [p.name for p in self.objective.varying_parameters()]
            # Add _merged suffix to get the merged samples
            var_names = [f"{name}_merged" for name in var_names]

        # Plot the results
        return az.plot_pair(
            data,
            var_names=var_names,
            kind=["scatter", "kde"],
            marginals=True,
            figsize=figsize,
        )


def _run_sampler(fitter, pool, sampler_kws, h, callback):
    """
    Run the sampler with appropriate settings.

    Parameters
    ----------
    fitter : Fitter
        The fitter instance
    pool : map-like object
        The pool for parallelization
    sampler_kws : dict
        Keywords to pass to the sampler
    h : file or None
        File handle for text-based checkpointing (ptemcee)
    callback : callable
        Callback function for each step
    """
    # if you're not creating more than 1 thread, then don't bother with a pool
    if isinstance(fitter.sampler, emcee.EnsembleSampler):
        if pool == 1:
            fitter.sampler.pool = None
        else:
            fitter.sampler.pool = pool

        # For emcee backend
        for state in fitter.sampler.sample(fitter._state, **sampler_kws):
            fitter._state = state
            if callback is not None:
                callback(state.coords, state.log_prob)
    else:
        # For ptemcee
        sampler_kws["mapper"] = pool

        # perform the sampling with text-based checkpointing
        for state in fitter.sampler.sample(fitter._state, **sampler_kws):
            fitter._state = state
            if h is not None:
                h.write(" ".join(map(str, state.coords.ravel())))
                h.write("\n")
            if callback is not None:
                callback(state.coords, state.log_prob)


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


def improved_dynamic_burnin(fitter, min_frac=0.1, threshold=0.05):
    """
    Improved burn-in detection using Geweke diagnostics and log-posterior trends.

    Parameters
    ----------
    fitter : Fitter object
        The fitter containing MCMC chains
    min_frac : float
        Minimum fraction of chain to consider (default: 0.1)
    threshold : float
        Threshold for Geweke Z-score

    Returns
    -------
    burnin : int
        Number of steps to discard as burn-in
    """
    try:
        import arviz as az
    except ImportError as err:
        e = (
            "ArviZ and xarray are required for this feature. "
            "Install them with 'pip install arviz xarray'"
        )
        raise ImportError(e) from err

    # Get chains data
    idata = fitter.to_arviz()

    # 1. Calculate Geweke diagnostics across all parameters
    geweke_stats = {}
    for var in idata.posterior.data_vars:
        if not var.endswith("_merged"):
            # Compute Geweke for each chain
            z_scores = []
            for chain in range(idata.posterior.sizes["chain"]):
                samples = idata.posterior[var].sel(chain=chain).values.flatten()
                z = az.geweke(samples, first=0.1, last=0.5)["z"]
                z_scores.append(np.abs(z))
            geweke_stats[var] = np.max(z_scores)  # Most conservative

    # 2. Examine log-posterior trend
    log_post = -np.array(fitter.sampler.get_log_prob())
    n_steps = log_post.shape[0]

    # Apply rolling mean to smooth log-posterior
    window_size = min(50, n_steps // 10)
    smoothed = np.array(
        [np.mean(log_post[max(0, i - window_size) : i + 1]) for i in range(n_steps)]
    )

    # Find where the derivative stabilizes
    derivatives = np.gradient(smoothed)
    stable_idx = np.where(np.abs(derivatives) < np.std(derivatives) * 0.1)[0]
    log_p_burnin = stable_idx[0] if len(stable_idx) > 0 else int(n_steps * min_frac)

    # 3. Find where parameters have converged (Geweke)
    geweke_burnins = []
    for var, max_z in geweke_stats.items():
        if max_z > threshold:
            # Not converged by Geweke - use conservative estimate
            geweke_burnins.append(int(n_steps * 0.5))
        else:
            # Find earliest point where Z-scores are below threshold
            for i in range(10, n_steps, 10):
                first_frac = i / n_steps
                if first_frac > 0.4:  # Don't get too aggressive
                    geweke_burnins.append(int(n_steps * 0.3))
                    break

                z_scores = []
                for chain in range(idata.posterior.sizes["chain"]):
                    samples = idata.posterior[var].sel(chain=chain).values.flatten()
                    z = az.geweke(samples, first=first_frac, last=0.5)["z"]
                    z_scores.append(np.abs(z))

                if np.max(z_scores) < threshold:
                    geweke_burnins.append(i)
                    break

    # Take maximum of all estimates with a safety margin
    burnin = int(max(max(geweke_burnins), log_p_burnin) * 1.1)
    return min(burnin, int(n_steps * 0.8))  # Cap at 80% of chain


def enhanced_convergence_check(
    fitter, min_ess=1000, max_rhat=1.01, stable_frac=0.85, logp_tol=0.01
):
    """
    Enhanced convergence check using multiple diagnostics.

    Parameters
    ----------
    fitter : Fitter
        The fitter object containing chains
    min_ess : float
        Minimum effective sample size required
    max_rhat : float
        Maximum acceptable R-hat statistic
    stable_frac : float
        Required fraction of parameters with stable distributions
    logp_tol : float
        Tolerance for log-posterior stability

    Returns
    -------
    bool
        Whether chains have converged
    dict
        Diagnostic statistics
    """
    import arviz as az

    # Convert to arviz format
    idata = fitter.to_arviz()

    # 1. Basic convergence diagnostics
    ess_stats = az.ess(idata)
    bulk_ess = ess_stats.ess_bulk.to_array().min().item()
    tail_ess = ess_stats.ess_tail.to_array().min().item()
    rhat_val = az.rhat(idata).to_array().max().item()

    # 2. Check log-posterior stability
    log_post = -np.array(fitter.sampler.get_log_prob())
    n_steps = log_post.shape[0]
    window_size = min(200, n_steps // 5)

    if n_steps > 2 * window_size:
        first_window = log_post[:window_size].mean()
        last_window = log_post[-window_size:].mean()
        logp_change = np.abs(
            (last_window - first_window) / (np.abs(first_window) + 1e-10)
        )
        logp_stable = logp_change < logp_tol
    else:
        logp_stable = False

    # 3. Check parameter stability using Geweke diagnostics
    stable_params = 0
    total_params = 0
    geweke_stats = {}

    for var in idata.posterior.data_vars:
        if var.endswith("_merged"):
            continue

        total_params += 1
        # Check stability for each chain
        chain_stable = True
        for chain in range(idata.posterior.sizes["chain"]):
            samples = idata.posterior[var].sel(chain=chain).values.flatten()

            # Skip if too few samples
            if len(samples) < 500:
                chain_stable = False
                break

            try:
                geweke = az.geweke(samples)
                # Check if Geweke z-scores are within ±2
                if np.any(np.abs(geweke["z"]) > 2):
                    chain_stable = False
            except Exception:
                chain_stable = False

        if chain_stable:
            stable_params += 1

        geweke_stats[var] = chain_stable

    # Final convergence assessment
    converged = (
        bulk_ess > min_ess
        and tail_ess > min_ess * 0.5  # Tail ESS can be lower
        and rhat_val < max_rhat
        and stable_params / max(1, total_params) >= stable_frac
        and logp_stable
    )

    # Return diagnostics along with convergence result
    diagnostics = {
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "rhat": rhat_val,
        "stable_fraction": stable_params / max(1, total_params),
        "logp_stable": logp_stable,
        "geweke": geweke_stats,
    }

    return converged, diagnostics


def adaptive_sampling(obj, target_ess=2000, max_samples=50000, initial_steps=1000):
    """
    Run MCMC with adaptive sampling until convergence criteria are met.

    Parameters
    ----------
    obj : Objective
        The objective function to sample
    target_ess : int
        Target effective sample size
    max_samples : int
        Maximum number of total samples to draw
    initial_steps : int
        Size of initial sampling batch
    """
    import time

    import arviz as az
    import matplotlib.pyplot as plt

    # Create fitter with HDF5 backend
    fitter = Fitter(obj)

    # Initial optimization to find good starting point
    print("Running initial optimization...")
    fitter.fit(
        method="differential_evolution",
        workers=-1,
        x0=[p.value for p in obj.varying_parameters()],
    )
    # Initialize with jitter around optimum
    from multiprocessing import Pool

    with Pool() as pool:
        fitter = Fitter(obj, pool=pool)
        ess_history, rhat_history, sampling_history = perform_sampling(
            target_ess, max_samples, initial_steps, time, fitter
        )

    # Get optimal burnin
    burnin = improved_dynamic_burnin(fitter)
    print(f"Recommended burn-in: {burnin} samples")

    # Final diagnostics
    idata = fitter.to_arviz(burn=burnin)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # ESS history
    axes[0].plot(sampling_history, ess_history)
    axes[0].axhline(target_ess, color="red", linestyle="--")
    axes[0].set_ylabel("Effective Sample Size")

    # R-hat history
    axes[1].plot(sampling_history, rhat_history)
    axes[1].axhline(1.01, color="red", linestyle="--")
    axes[1].set_ylabel("R-hat")

    # Log-probability
    log_probs = -np.array(fitter.sampler.get_log_prob())[burnin:]
    axes[2].plot(range(len(log_probs)), log_probs, alpha=0.3, color="gray")
    axes[2].plot(
        range(len(log_probs)),
        np.convolve(log_probs, np.ones(50) / 50, mode="same"),
        color="blue",
    )
    axes[2].set_ylabel("Log Probability")
    axes[2].set_xlabel("Iteration (post-burnin)")

    plt.tight_layout()
    plt.savefig("sampling_diagnostics.png")

    # Create trace plots
    az.plot_trace(idata)
    plt.savefig("trace_plots.png")

    # Generate summary
    summary = az.summary(idata)

    # Save results
    results = {
        "idata": idata,
        "obj": obj,
        "fitter": fitter,
        "burnin": burnin,
        "summary": summary,
        "convergence_stats": {
            "ess_history": ess_history,
            "rhat_history": rhat_history,
            "sampling_history": sampling_history,
        },
    }

    with Path(f"{obj.name}_mcmc_results.pkl").open("wb") as f:
        import pickle

        pickle.dump(results, f)

    return fitter, burnin


def perform_sampling(target_ess, max_samples, initial_steps, time, fitter):
    """
    Perform the sampling process with adaptive batch sizes.
    """
    fitter.initialise("jitter")

    # Track diagnostics over time
    ess_history = []
    rhat_history = []
    logp_history = []
    sampling_history = []

    # Main sampling loop
    total_samples = 0
    batch_size = initial_steps
    start_time = time.time()

    while total_samples < max_samples:
        print(f"Sampling batch of {batch_size} steps (total so far: {total_samples})")

        # Run batch of samples
        fitter.sample(batch_size, use_checkpoint=True, verbose=True)
        total_samples += batch_size

        # Check convergence
        converged, stats = enhanced_convergence_check(fitter, min_ess=target_ess)

        # Record diagnostics
        ess_history.append(stats["bulk_ess"])
        rhat_history.append(stats["rhat"])
        logp_history.append(stats.get("logp_stable", False))
        sampling_history.append(total_samples)

        # Print diagnostics
        elapsed = time.time() - start_time
        print(f"Diagnostics after {total_samples} samples ({elapsed:.1f}s):")
        print(f"  - ESS: {stats['bulk_ess']:.1f}/{target_ess}")
        print(f"  - R-hat: {stats['rhat']:.4f}")
        print(f"  - Stable params: {stats['stable_fraction']:.2f}")
        print(f"  - Log-P stable: {stats['logp_stable']}")

        # Exit if converged
        if converged:
            print(f"Convergence achieved after {total_samples} samples!")
            break

        # Adjust batch size based on progress
        if stats["bulk_ess"] < target_ess * 0.1:
            # Far from target, use larger batches
            batch_size = min(5000, max_samples - total_samples)
        elif stats["bulk_ess"] < target_ess * 0.5:
            # Getting closer, medium batches
            batch_size = min(2000, max_samples - total_samples)
        else:
            # Near target, smaller batches
            batch_size = min(1000, max_samples - total_samples)

        # If we're out of samples, break
        if batch_size <= 0:
            print(f"Reached maximum samples ({max_samples}) without convergence")
            break
    return ess_history, rhat_history, sampling_history
