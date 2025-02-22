"""XRR reflectivity model for the package."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Parameters, possibly_create_parameter
from refnx.dataset import ReflectDataset
from scipy.interpolate import splev, splrep

from pyref.fitting.uniaxial_model import uniaxial_reflectivity

if TYPE_CHECKING:
    import polars as pl

# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))


class XrayReflectDataset(ReflectDataset):
    """Overload of the ReflectDataset class from refnx."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        diff = np.diff(self.x)
        # locate where diff is less than 0 and find that index
        idx = np.where(diff < 0)[0] + 1
        if len(idx) > 0:
            self.s = ReflectDataset(
                (self.x[: idx[0]], self.y[: idx[0]], self.y_err[: idx[0]])
            )
            self.p = ReflectDataset(
                (self.x[idx[0] :], self.y[idx[0] :], self.y_err[idx[0] :])
            )
        else:
            self.s = ReflectDataset((self.x, self.y, self.y_err))
            self.p = ReflectDataset((self.x, self.y, self.y_err))

        # calculate the anisotropic ratio in the overlaping region
        q_min = np.max([self.s.x.min(), self.p.x.min()])
        q_max = np.min([self.s.x.max(), self.p.x.max()])
        q_common = np.linspace(q_min, q_max, 100)

        # Interpolate both s and p polarized data onto common q points
        r_s_interp = np.interp(q_common, self.s.x, self.s.y)
        r_p_interp = np.interp(q_common, self.p.x, self.p.y)

        _anisotropy = (r_p_interp - r_s_interp) / (r_p_interp + r_s_interp)
        self.anisotropy = ReflectDataset((q_common, _anisotropy))

    def plot(self, ax=None, ax_anisotropy=None, **kwargs):
        """Plot the reflectivity and anisotropy data."""
        if ax is None:
            fig, axs = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax = axs[0]
            ax_anisotropy = axs[1]

        if self.s.y[3] != self.p.y[3]:
            # Plot s and p separately
            ax.errorbar(
                self.s.x,
                self.s.y,
                self.s.y_err,
                label=f"{self.name} s-pol",
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )
            ax.errorbar(
                self.p.x,
                self.p.y,
                self.p.y_err,
                label=f"{self.name} p-pol",
                marker="o",
                color="C1",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )
        else:
            # Plot together if same x values
            ax.errorbar(
                self.x,
                self.y,
                self.y_err,
                label=self.name,
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )

        ax_anisotropy.plot(
            self.anisotropy.x,
            self.anisotropy.y,
            label=f"{self.name} anisotropy" if self.name else "anisotropy",
            marker="o",
            markersize=3,
            lw=0,
            color="C2",
        )
        ax_anisotropy.axhline(
            0,
            color=plt.rcParams["axes.edgecolor"],
            ls="-",
            lw=plt.rcParams["axes.linewidth"],
        )

        ax.set_yscale("log")
        ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")
        ax.set_ylabel(r"$R$")
        plt.legend()
        return ax, ax_anisotropy


def to_reflect_dataset(
    df: pl.DataFrame, *, overwrite_err: bool = True
) -> XrayReflectDataset:
    """Convert a pandas dataframe to a ReflectDataset object."""
    if not overwrite_err:
        e = "overwrite_err=False is not implemented yet."
        raise NotImplementedError(e)
    Q = df["Q"].to_numpy()
    R = df["r"].to_numpy()
    # Calculate initial dR
    dR = 0.15 * R + 0.3e-6 * Q
    # Ensure dR doesn't exceed 90% of R to keep R-dR positive
    dR = np.minimum(dR, 0.9 * R)
    ds = XrayReflectDataset(data=(Q, R, dR))
    return ds


class ReflectModel:
    r"""
    Reflectometry model for anisotropic interfaces.

    Parameters
    ----------
    structure : anisotropic_structure.PXR_Structure object
        The interfacial structure.
    scale : float or refnx.analysis.Parameter, optional
        scale factor. All model values are multiplied by this value before
        the background is added. This is turned into a Parameter during the
        construction of this object.
    bkg : float or refnx.analysis.Parameter, optional
        Q-independent constant background added to all model values. This is
        turned into a Parameter during the construction of this object.
    name : str, optional
        Name of the Model
    dq : float or refnx.analysis.Parameter, optional

        - `dq == 0` then no resolution smearing is employed.
        - `dq` is a float or refnx.analysis.Parameter
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.This value is turned into a Parameter during the
           construction of this object.

    Notes
    -----
    If `x_err` is supplied to the `model` method, dq becomes overriden. that
    overrides any setting given here.

    Adding q-smearing greatly reduces the current speed of the calculation.
    Data collected at ALS 11.0.1.2 over the carbon edge likely does not require any
    q-smearing.

    """

    def __init__(
        self,
        structure,
        energy: float | None = None,
        pol: Literal["s", "p", "sp", "ps"] = "s",
        name="",
        *,
        scale_s=1,
        scale_p=1,
        bkg=0,
        dq=0.0,
        q_offset=0.0,
        en_offset=0.0,
        theta_offset_s=0.0,
        theta_offset_p=0.0,
        phi=0,
        backend="uni",
    ):
        self.name = name
        self._parameters = None
        self.backend = backend
        self._energy = energy  # [eV]
        self._phi = phi
        self._pol = pol  # Output polarization

        # all reflectometry models have an optional scale factor and background
        self._scale_s = possibly_create_parameter(scale_s, name="scale_s")
        self._scale_p = possibly_create_parameter(scale_p, name="scale_p")

        self._bkg = possibly_create_parameter(bkg, name="bkg")
        self._q_offset = possibly_create_parameter(q_offset, name="q_offset")
        self._theta_offset = possibly_create_parameter(0, name="theta_offset")

        # New model parameter en_offset : 10/21/2021
        self._en_offset = possibly_create_parameter(en_offset, name="en_offset")

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name="dq - resolution")

        # New model parameters for theta_offset
        self._theta_offset_s = possibly_create_parameter(
            theta_offset_s, name="theta_offset_s"
        )
        self._theta_offset_p = possibly_create_parameter(
            theta_offset_p, name="theta_offset_p"
        )

        self._structure = None
        self.structure = structure

    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model.

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.


        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity


        Note:
        -------
        Uses the assigned 'Pol' to determine the output state of 's-pol', 'p-pol' or
        both
        """
        return self.model(x, p=p, x_err=x_err)

    def __repr__(self):
        """Representation of the ReflectModel."""
        return (
            "ReflectModel({_structure!r}, name={name!r},"
            " scale={_scale!r}, bkg={_bkg!r},"
            " dq={_dq!r}"
            " quad_order={quad_order}),"
            " q_offset={_q_offset!r}".format(**self.__dict__)
        )

    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`.

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq

    @dq.setter
    def dq(self, value):
        self._dq.value = value

    @property
    def scale_s(self):
        r"""

        :class:`refnx.analysis.Parameter`.

          - all model values are multiplied by this value before the background is
          added.

        """
        return self._scale_s

    @scale_s.setter
    def scale_s(self, value):
        self._scale_s.value = value

    @property
    def scale_p(self):
        r"""

        :class:`refnx.analysis.Parameter`.

          - all model values are multiplied by this value before the background is
          added.

        """
        return self._scale_p

    @scale_p.setter
    def scale_p(self, value):
        self._scale_p.value = value

    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter`.

          - linear background added to all model values.
        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value

    @property
    def q_offset(self):
        r"""
        :class:`refnx.analysis.Parameter`.

          - offset in q-vector due to experimental error
        """
        return self._q_offset

    @q_offset.setter
    def q_offset(self, value):
        self._q_offset.value = value

    @property
    def theta_offset_s(self):
        r"""
        :class:`refnx.analysis.Parameter`.

          - offset in theta for s-polarization due to experimental error
        """
        return self._theta_offset_s

    @theta_offset_s.setter
    def theta_offset_s(self, value):
        self._theta_offset_s.value = value

    @property
    def theta_offset_p(self):
        r"""
        :class:`refnx.analysis.Parameter`.

          - offset in theta for p-polarization due to experimental error
        """
        return self._theta_offset_p

    @theta_offset_p.setter
    def theta_offset_p(self, value):
        self._theta_offset_p.value = value

    @property
    def en_offset(self):
        r"""
        :class:`refnx.analysis.Parameter`.

        - offset in q-vector due to experimental error

        """
        return self._en_offset

    @en_offset.setter
    def en_offset(self, value):
        self._en_offset.value = value

    @property
    def energy(self):
        """
        Photon energy to evaluate the resonant reflectivity.

        Automatically updates all PXR_MaterialSLD objects associated with
        self.structure.

        Returns
        -------
            energy : float
                Photon energy of X-ray probe.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy

    @property
    def pol(self):
        """
        Polarization to calculate the resonant reflectivity.

            -`pol == 's'`
            Calculation returns s-polarization only.
            -`pol == 'p'`
            Calculation returns p-polarization only.
            -`pol == 'sp' or 'ps'`
            Calulation returns concatenate in order of input.

        Returns
        -------
            pol : str
                Linear polarizations state of incident raw
        """
        return self._pol

    @pol.setter
    def pol(self, pol):
        self._pol = pol

    @property
    def phi(self):
        """
        Azimuthal angle of incidence [deg]. Only used with a biaxial calculation.

        Returns
        -------
            phi : float
                Azimuthal angle of incidence used in calculation.
        """
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

    def _model(self, x, p=None, x_err=None):
        if p is not None:
            self.parameters.pvals = np.array(p)

        if x_err is None:
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        # Multipol fitting is currently done through concatenating
        # s- and p-pol together.
        # A temp x-data set is used to calculate the model based on the q-range
        if self.pol == "sp" or self.pol == "ps":
            concat_loc = np.argmax(
                np.abs(np.diff(x))
            )  # Location where the q-range swaps for high s-pol to low p-pol
            qvals_1 = x[: concat_loc + 1]  # Split inputs for later
            qvals_2 = x[concat_loc + 1 :]  # Split inputs for later
            num_q = (
                concat_loc + 50
            )  # 50 more points to make sure the interpolation works
            qvals = np.linspace(np.min(x), np.max(x), num_q)

            # Convert q to theta for offset application
            theta_s = (
                np.arcsin(qvals_1 * 12398.42 / (4 * np.pi * self.energy)) * 180 / np.pi
            )
            theta_p = (
                np.arcsin(qvals_2 * 12398.42 / (4 * np.pi * self.energy)) * 180 / np.pi
            )

            # Apply offsets
            theta_s += self.theta_offset_s.value
            theta_p += self.theta_offset_p.value

            # Convert back to q
            qvals_1 = 4 * np.pi * self.energy * np.sin(theta_s * np.pi / 180) / 12398.42
            qvals_2 = 4 * np.pi * self.energy * np.sin(theta_p * np.pi / 180) / 12398.42
            # For single polarization, apply appropriate offset
        elif self.pol == "s":
            theta = np.arcsin(x * 12398.42 / (4 * np.pi * self.energy)) * 180 / np.pi
            theta += self.theta_offset_s.value
            qvals = 4 * np.pi * self.energy * np.sin(theta * np.pi / 180) / 12398.42
        elif self.pol == "p":
            theta = np.arcsin(x * 12398.42 / (4 * np.pi * self.energy)) * 180 / np.pi
            theta += self.theta_offset_p.value
            qvals = 4 * np.pi * self.energy * np.sin(theta * np.pi / 180) / 12398.42
        else:
            qvals = x
            qvals = x

        refl, tran, *components = reflectivity(
            qvals + self.q_offset.value,
            self.structure.slabs(),
            self.structure.tensor(energy=self.energy),
            self.energy,
            self.phi,
            scale_s=self.scale_s.value,
            scale_p=self.scale_p.value,
            bkg=self.bkg.value,
            dq=x_err,
            backend=self.backend,
        )

        return qvals, qvals_1, qvals_2, refl, tran, components

    def model(self, x, p=None, x_err=None):
        r"""
        Calculate the reflectivity of this model.

        Parameters
        ----------
        x : float or np.ndarray
            q or E values for the calculation.
            specifiy self.qval to be any value to fit energy-space
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity. Output is dependent on `self.pol`

        """
        qvals, qvals_1, qvals_2, refl, tran, components = self._model(x, p, x_err)

        # Return result based on desired polarization:

        if self.pol == "s":
            output = refl[:, 1, 1]
        elif self.pol == "p":
            output = refl[:, 0, 0]
        elif self.pol == "sp":
            spol_model = np.interp(qvals_1, qvals, refl[:, 1, 1])
            ppol_model = np.interp(qvals_2, qvals, refl[:, 0, 0])
            output = np.concatenate([spol_model, ppol_model])
        elif self.pol == "ps":
            spol_model = np.interp(qvals_2, qvals, refl[:, 1, 1])
            ppol_model = np.interp(qvals_1, qvals, refl[:, 0, 0])
            output = np.concatenate([ppol_model, spol_model])

        else:
            print("No polarizations were chosen for model")
            output = 0

        return output

    def anisotropy(self, x, p=None, x_err=None):
        """Calculate the anisotropy of the model."""
        q_vals, qvals_1, qvals_2, refl, tran, components = self._model(x, p, x_err)

        q_min = np.max([qvals_1.min(), qvals_2.min()])
        q_max = np.min([qvals_1.max(), qvals_2.max()])

        q_common = np.linspace(q_min, q_max, 100)

        r_s = np.interp(q_common, q_vals, refl[:, 1, 1])
        r_p = np.interp(q_common, q_vals, refl[:, 0, 0])

        return q_common, (r_p - r_s) / (r_p + r_s)

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model.

        Do not
        include log-probability terms for model parameters, these are
        automatically included elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        return self.structure.logp()

    @property
    def structure(self):
        r"""
        :class:`PRSoXR.PXR_Structure`.

           - object describing the interface of a reflectometry sample.
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        p = Parameters(name="instrument parameters")
        p.extend(
            [
                self.scale_s,
                self.scale_p,
                self.bkg,
                self.dq,
                self.q_offset,
                self.en_offset,
                self.theta_offset_s,
                self.theta_offset_p,
            ]
        )

        self._parameters = Parameters(name=self.name)
        self._parameters.extend([p, structure.parameters])

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`.

           - parameters associated with this model.
        """
        self.structure = self._structure
        return self._parameters


def reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy: float = 250.0,
    phi: float = 0,
    scale_s: float = 1.0,
    scale_p: float = 1.0,
    bkg: float = 0.0,
    dq: float = 0.0,
    theta_offset: float = 0.0,
    backend: Literal["uni", "bi"] = "uni",
):
    r"""
    Full calculation for anisotropic reflectivity of a stratified medium.

    Parameters
    ----------
     q : np.ndarray
         The qvalues required for the calculation.
         :math:`Q=\frac{4Pi}{\lambda}\sin(\Omega)`.
         Units = Angstrom**-1
     slabs : np.ndarray
         coefficients required for the calculation, has shape (2 + N, 4),
         where N is the number of layers

         - slabs[0, 0]
            ignored
         - slabs[N, 0]
            thickness of layer N
         - slabs[N+1, 0]
            ignored

         - slabs[0, 1]
            trace of real index tensor of fronting (/1e-6 Angstrom**-2)
         - slabs[N, 1]
            trace of real index tensor of layer N (/1e-6 Angstrom**-2)
         - slabs[-1, 1]
            trace of real index tensor of backing (/1e-6 Angstrom**-2)

         - slabs[0, 2]
            trace of imaginary index tensor of fronting (/1e-6 Angstrom**-2)
         - slabs[N, 2]
            trace of imaginary index tensor of layer N (/1e-6 Angstrom**-2)
         - slabs[-1, 2]
            trace of imaginary index tensor of backing (/1e-6 Angstrom**-2)

         - slabs[0, 3]
            ignored
         - slabs[N, 3]
            roughness between layer N-1/N
         - slabs[-1, 3]
            roughness between backing and layer N

     tensor : 3x3 numpy array
         The full dielectric tensor required for the anisotropic calculation.
         Each component (real and imaginary) is a fit parameter
         Has shape (2 + N, 3, 3)
         units - unitless

     energy : float
         Energy to calculate the reflectivity profile
         Used in calculating 'q' and index of refraction for PXR_MaterialSLD objects

     phi : float
         Azimuthal angle of incidence for calculating k-vectors.
         This is only required if dealing with a biaxial tensor
         defaults to phi = 0 ~

     scale : float
         scale factor. All model values are multiplied by this value before
         the background is added

     bkg : float
         Q-independent constant background added to all model values.

     dq : float or np.ndarray, optional
         - `dq == 0`
            no resolution smearing is employed.
         - `dq` is a float
            a constant dQ/Q resolution smearing is employed.  For 5% resolution
            smearing supply 5.

    backend : str ('uni' or 'bi')
         Calculation symmetry to be applied. 'uni' for a uniaxial approximation
         (~10x increase in speed). 'bi' for full biaxial calculation.

    Example
    -------

    from refnx.reflect import reflectivity
    ```python
    q = np.linspace(0.01, 0.5, 1000)
    slabs = np.array(
        [
            [0, 2.07, 0, 0],
            [100, 3.47, 0, 3],
            [500, -0.5, 0.00001, 3],
            [0, 6.36, 0, 3],
        ]
    )
    print(reflectivity(q, slabs))
    ```
    """
    # constant dq/q smearing
    if isinstance(dq, numbers.Real):
        if float(dq) == 0:
            if backend == "uni":
                refl, tran, *components = uniaxial_reflectivity(
                    q, slabs, tensor, energy
                )
            else:
                refl, tran, *components = uniaxial_reflectivity(
                    q, slabs, tensor, energy, phi
                )
            # Scale s and p polarizations separately
            refl[:, 0, 0] = scale_s * refl[:, 0, 0]
            refl[:, 1, 1] = scale_p * refl[:, 1, 1]
            return (refl + bkg), tran, components
        else:
            smear_refl, smear_tran, *components = _smeared_reflectivity(
                q, slabs, tensor, energy, phi, dq, backend=backend
            )
            # Scale s and p polarizations separately
            smear_refl[:, 0, 0] = scale_s * smear_refl[:, 0, 0]
            smear_refl[:, 1, 1] = scale_p * smear_refl[:, 1, 1]
            return (smear_refl + bkg), smear_tran, components

    return None


def _smeared_reflectivity(q, w, tensor, energy, phi, resolution, backend="uni"):
    """
    Fast resolution smearing for constant dQ/Q.

    Parameters
    ----------
    q: np.ndarray
        Q values to evaluate the reflectivity at
    w: np.ndarray
        Parameters for the reflectivity model
    resolution: float
        Percentage dq/q resolution. dq specified as FWHM of a resolution
        kernel.

    Returns
    -------
    reflectivity: np.ndarray
        The resolution smeared reflectivity
    """
    if resolution < 0.5:
        if backend == "uni":
            return uniaxial_reflectivity(q, w, tensor, energy)
        else:
            return uniaxial_reflectivity(q, w, tensor, energy, phi)
            # return yeh_4x4_reflectivity(q, w, tensor, energy, phi)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1.0 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(
        np.abs(1 * (np.abs(start - finish)) / (1.7 * resolution / _FWHM / gaussgpoint))
    )
    xtemp = np.linspace(start, finish, int(interpnum))
    xlin = np.power(10.0, xtemp)

    # resolution smear over [-4 sigma, 4 sigma]
    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)
    if backend == "uni":
        refl, tran, *components = uniaxial_reflectivity(xlin, w, tensor, energy)
    else:
        refl, tran, *components = uniaxial_reflectivity(xlin, w, tensor, energy)
    # Refl, Tran = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,
    # save_components=None)
    # Convolve each solution independently
    smeared_ss = np.convolve(refl[:, 0, 0], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_pp = np.convolve(refl[:, 1, 1], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_sp = np.convolve(refl[:, 0, 1], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_ps = np.convolve(refl[:, 1, 0], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )

    # smeared_rvals *= gauss_x[1] - gauss_x[0]

    # interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)
    #
    # smeared_output = interpolator(q)
    # Re-interpolate and organize the results wave following spline interpolation
    tck_ss = splrep(xlin, smeared_ss)
    smeared_output_ss = splev(q, tck_ss)

    tck_sp = splrep(xlin, smeared_sp)
    smeared_output_sp = splev(q, tck_sp)

    tck_ps = splrep(xlin, smeared_ps)
    smeared_output_ps = splev(q, tck_ps)

    tck_pp = splrep(xlin, smeared_pp)
    smeared_output_pp = splev(q, tck_pp)

    # Organize the output wave with the appropriate outputs
    smeared_output = np.rollaxis(
        np.array(
            [
                [smeared_output_ss, smeared_output_sp],
                [smeared_output_ps, smeared_output_pp],
            ]
        ),
        2,
        0,
    )

    return smeared_output, tran, components
