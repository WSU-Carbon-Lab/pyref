"""PXR Structure and Components."""

from __future__ import annotations

import operator
from collections import UserList

import numpy as np
import pandas as pd
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from refnx.reflect.interface import Erf, Step
from refnx.reflect.structure import Component
from scipy.interpolate import interp1d

from pyref.fitting.reflectivity import reflectivity

speed_of_light = 299792458  # m/s
plank_constant = 4.135667697e-15  # ev*s
hc = (speed_of_light * plank_constant) * 1e10  # ev*A

tensor_index = ["xx", "yy", "zz"]  # Indexing for later definitions

# azimuthal rotations for averaging
t90 = np.pi / 2
t180 = np.pi
t270 = 3 * np.pi / 2


class Structure(UserList):
    r"""
    Represents the interfacial Structure of a reflectometry sample.

    Successive Components are added to the Structure to construct the interface.

    Parameters
    ----------
    components : sequence
        A sequence of PXR_Components to initialise the PXR_Structure.
    name : str
        Name of this structure
    reverse_structure : bool
        If `Structure.reverse_structure` is `True` then  slab representation produced by
          `Structure.slabs` is reversed.

    Example
    -------
    >>> from PyPXR import PXR_SLD, PXR_MaterialSLD
    >>> en = 284.4  # [eV]
    >>> # make the material with tensor index of refraction
    >>> vac = PXR_MaterialSLD("", density=1, energy=en, name="vacuum")  # Superstrate
    >>> si = PXR_MaterialSLD("Si", density=2.33, energy=en, name="Si")  # Substrate
    >>> sio2 = PXR_MaterialSLD("SiO2", density=2.4, energy=en, name="SiO2")  # Substrate
    >>> n_xx = complex(-0.0035, 0.0004)  # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009)  # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name="material")  # molecule
    >>> # Make the structure
    >>> # See 'PXR_Slab' for details on building layers
    >>> structure = vac(0, 0) | molecule(100, 2) | sio2(15, 1.5) | si(1, 1.5)

    """

    def __init__(
        self, components=(), name="", reverse_structure=False
    ):  # Removed solvent parameter
        super().__init__()
        self._name = name

        self._reverse_structure = bool(reverse_structure)

        # if you provide a list of components to start with, then initialise
        # the structure from that
        self.data = [c for c in components if isinstance(c, Component)]

    def __copy__(self):
        """Create a shallow copy of the structure."""
        s = Structure(name=self.name)
        s.data = self.data.copy()
        return s

    def __setitem__(self, i, v):
        """Set the i-th item of the structure to v."""
        self.data[i] = v

    def __str__(self):
        """Representation of the structure."""
        s = []
        s.append("{:_>80}".format(""))
        s.append(f"Structure: {self.name!s: ^15}")
        s.append(f"reverse structure: {self.reverse_structure!s}")

        for component in self:
            s.append(str(component))

        return "\n".join(s)

    def __repr__(self):
        """Representation of the structure."""
        return (
            "Structure(components={data!r},"
            " name={_name!r},"
            " reverse_structure={_reverse_structure},".format(**self.__dict__)
        )

    def append(self, item):
        """
        Append a :class:`PXR_Component` to the Structure.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, Scatterer):
            self.append(item())
            return

        if not isinstance(item, PXR_Component):
            e = "You can only add PXR_Component objects to a structure"
            raise TypeError(e)
        super().append(item)

    @property
    def name(self):
        """Name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def reverse_structure(self):
        """
        Boolean flag for reversing the structure.

        **bool**  if `True` then the slab representation produced by
        :meth:`PXR_Structure.slabs` is reversed. The sld profile and calculated
        reflectivity will correspond to this reversed structure.
        """
        return bool(self._reverse_structure)

    @reverse_structure.setter
    def reverse_structure(self, reverse_structure):
        self._reverse_structure = reverse_structure

    def slabs(self):
        r"""
        Slab representation of the structure.

        Returns
        -------
        slabs : :class:`np.ndarray`
            Slab representation of this structure.
            Has shape (N, 3).
            N - number of slabs

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               sld.delta of layer N
            - slab[N, 2]
               sld.beta of layer N
            - slab[N, 3]
               roughness between layer N and N-1

        Notes
        -----
        If `PXR_Structure.reversed is True` then the slab representation order is
        reversed.
        """
        if not len(self):
            return None

        if not (isinstance(self.data[-1], Slab) and isinstance(self.data[0], Slab)):
            e = "The first and last PXR_Components in a PXR_Structure need to be PXR_slabs"  # noqa: E501
            raise TypeError(e)

        sl = [
            c.slabs(structure=self) for c in self.components
        ]  # concatenate PXR_Slab objects
        try:
            slabs = np.concatenate(sl)
        except ValueError:
            # some of slabs may be None. np can't concatenate arr and None
            slabs = np.concatenate([s for s in sl if s is not None])

        # if the slab representation needs to be reversed.
        reverse = self.reverse_structure
        if reverse:
            roughnesses = slabs[1:, 3]
            slabs = np.flipud(slabs)
            slabs[1:, 3] = roughnesses[::-1]
            slabs[0, 3] = 0.0

        return slabs

    def tensor(self, energy=None):
        """
        Tensor index of refraction for the structure.

        Parameters
        ----------
        energy: float
            Photon energy used to calculate the tensor index of refraction.
            This only applies for objects that require a specific energy
            (see PXR_MaterialSLD). Common for substrates/superstrates

        Returns
        -------
        tensors : :class:`np.ndarray`
            Supplementary object to self.slabs that contains dielectric tensor for each
            layer.
            Has shape (N, 3,3).
            N - number of slabs

            - tensor[N, 1, 1]
               dielectric component xx of layer N
            - tensor[N, 2, 2]
               dielectric component yy of layer N
            - tensor[N, 3, 3]
               dielectric component zz of layer N

        Notes
        -----
        Output as a (3, 3) np.ndarray.
        Used for broadcasting in later calculations. All off-diagonal elements are zero.

        If `Structure.reversed is True` then the representation order is
        reversed. Energy is required for energy-dependent slabs

        """
        d1 = [c.tensor(energy=energy) for c in self.components]
        try:
            _tensor = np.concatenate(d1, axis=0)
        except ValueError:
            # some of slabs may be None. np can't concatenate arr and None
            _tensor = np.concatenate([s for s in d1 if s is not None], axis=0)

        reverse = self.reverse_structure
        if reverse:
            _tensor = np.flip(_tensor, axis=0)
        return _tensor

    def reflectivity(self, q, energy=250.0, backend="uni"):
        """
        Calculate theoretical polarized reflectivity of this structure.

        Parameters
        ----------
        q : array-like
            Q values (Angstrom**-1) for evaluation
        energy : float
            Photon energy (eV) for evaluation
        backend : 'uni' or 'biaxial'
            Specifies if you want to run a uniaxial calculation or a full
            biaxial calculation. Biaxial has NOT been verified through outside means
            (07/2021 Biaxial currently does not work)

        """
        refl, tran, *components = reflectivity(
            q, self.slabs(), self.tensor(energy=energy), backend=backend
        )
        return refl[:, 1, 1], refl[:, 0, 0], components

    def sld_profile(self, z=None, align=0):
        """
        Create an index of refraction depth profile.

        Parameters
        ----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the fronting
            medium and first layer.
        align : int, optional
            Places a specified interface in the slab representation of a PXR_Structure
            at z =0. Python indexing is allowed to select interface.

        Returns
        -------
        zed : np.ndarray
            Interfacial distance measured from superstrate offset by 'align'.
            Has shape (N, )
        prof : np.ndarray (complex)
            Real and imaginary tensor components of index of refraction [unitless]
            Has shape (N, 3)

            -prof[N, 0]
                dielectric component n_xx at depth N
            -prof[N, 1]
                dielectric component n_yy at depth N
            -prof[N, 3]
                dielectric component n_xx at depth N

        Notes
        -----
        >>> # To calculate the isotropic components
        >>> n_iso = prof.sum(axis=1) / 3  # (nxx + nyy + nzz)/3
        >>> # To calculate the birefringence and dichroism
        >>> diff = prof[:, 0] - prof[:, 2]  # nxx-nzz

        """
        slabs = self.slabs()
        tensor = self.tensor()
        if (
            (slabs is None)
            or (len(slabs) < 2)
            or (not isinstance(self.data[0], Slab))
            or (not isinstance(self.data[-1], Slab))
        ):
            e = "Structure requires fronting and backing Slabs in order to calculate."
            raise TypeError(e)

        zed, prof = birefringence_profile(slabs, tensor, z)

        offset = 0
        if align != 0:
            align = int(align)
            if align >= len(slabs) - 1 or align < -1 * len(slabs):
                e = "abs(align) has to be less than len(slabs) - 1"
                raise RuntimeError(e)
            # to figure out the offset you need to know the cumulative distance
            # to the interface
            slabs[0, 0] = slabs[-1, 0] = 0.0  # Set the thickness of each end to zero
            if align >= 0:
                offset = np.sum(slabs[: align + 1, 0])
            else:
                offset = np.sum(slabs[:align, 0])
        return zed - offset, prof

    def __ior__(self, other):
        """
        Build a structure by `IOR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`PXR_Structure`, :class:`PXR_Component`, :class:`PXR_SLD`
            The object to add to the structure.

        Examples
        --------
        ```python
        air = SLD(0, name="air")
        sio2 = SLD(3.47, name="SiO2")
        si = SLD(2.07, name="Si")
        structure = air | sio2(20, 3)
        structure |= si(0, 4)
        ```
        """
        # self |= other
        if isinstance(other, PXR_Component):
            self.append(other)
        elif isinstance(other, Structure):
            self.extend(other.data)
        elif isinstance(other, Scatterer):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise TypeError()

        return self

    def __or__(self, other: Structure | Component | SLD) -> Structure:
        """
        Build a structure by `OR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`PXR_Structure`, :class:`PXR_Component`, :class:`PXR_SLD`
            The object to add to the structure.

        Examples
        --------
        ```python
        vac = PXR_MaterialSLD("", density=1, energy=en, name="vacuum")  # Superstrate
        sio2 = PXR_MaterialSLD("SiO2", density=2.4, energy=en, name="SiO2")  # Substrate
        si = PXR_MaterialSLD("Si", density=2.33, energy=en, name="Si")  # Substrate
        structure = vac | sio2(10, 5) | si(0, 1.5)
        ```
        """
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    @property
    def components(self):
        """
        The list of components in the sample.
        """
        return self.data

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`.

        all the parameters associated with this structure.
        """
        p = Parameters(name=f"Structure - {self.name}")
        p.extend([component.parameters for component in self.components])
        return p

    def logp(self):
        """
        log-probability for the interfacial structure.

        Note that if a given
        component is present more than once in a Structure then it's log-prob
        will be counted twice.

        Returns
        -------
        logp : float
            log-prior for the Structure.
        """
        logp = 0
        for component in self.components:
            logp += component.logp()

        return logp

    def plot(
        self,
        pvals=None,
        samples=0,
        ax=None,
        difference=False,
        align=0,
    ):
        """
        Plot the structure.

        Requires matplotlib be installed.

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying
        samples: number
            If this structures constituent parameters have been sampled, how
            many samples you wish to plot on the graph.
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.
        difference: boolean, optional
            If True, plot the birefringence / dichroism on a separate graph.
        align: int, optional
            Aligns the plotted structures around a specified interface in the
            slab representation of a Structure. This interface will appear at
            z = 0 in the sld plot.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
          `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        params = self.parameters

        if pvals is not None:
            params.pvals = pvals

        if ax is None:
            _, ax = plt.subplots()

        if samples > 0:
            saved_params = np.array(params)
            # Get a number of chains, chosen randomly, and plot the model.
            for pvec in self.parameters.pgen(ngen=samples):
                params.pvals = pvec

                temp_zed, temp_prof = self.sld_profile(align=align)
                temp_iso = temp_prof.sum(axis=1) / 3  # (nxx + nyy + nzz)/3
                ax.plot(temp_zed, temp_iso, color="k", alpha=0.01)

            # put back saved_params
            params.pvals = saved_params

        # parameters to plot
        zed, prof = self.sld_profile(align=align)
        iso = prof.sum(axis=1) / 3
        ax.plot(zed, np.real(iso), color="C0", zorder=20, label="δ", linewidth=0.9)
        ax.plot(
            zed,
            np.real(prof[:, 0]),
            color="C0",
            zorder=10,
            label="δxx",
            linestyle="--",
        )
        ax.plot(
            zed,
            np.real(prof[:, 2]),
            color="C0",
            zorder=10,
            label="δzz",
            linestyle=":",
        )
        ax.plot(zed, np.imag(iso), color="C2", zorder=20, label="β", linewidth=0.9)
        ax.plot(
            zed,
            np.imag(prof[:, 0]),
            color="C2",
            zorder=10,
            label="βxx",
            linestyle="--",
        )
        ax.plot(
            zed,
            np.imag(prof[:, 2]),
            color="C2",
            zorder=10,
            label="βzz",
            linestyle=":",
        )
        # ax.plot(*self.sld_profile(align=align), color='red', zorder=20)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Index of refraction")
        ax.set_xlabel("zed / $\\AA$")
        if difference:
            axr = ax.twinx()
            dichroism = prof[:, 0].real - prof[:, 2].real
            axr.plot(zed, dichroism, color="C1", zorder=20)
            axr.fill_between(
                zed, dichroism, color="C1", alpha=0.5, zorder=20, label="δxx - δzz"
            )
            axr.set_ylabel("δxx - δzz")
            axr.tick_params(axis="y", labelcolor="C1", color="C1")
            axr.spines["right"].set_color("C1")
            axr.set_ylim(ax.get_ylim())
            axr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axr.legend()
        ax.legend()
        return ax


class Scatterer:
    """
    Abstract base class for a material with a complex tensor index of refraction.
    """

    def __init__(self, name=""):
        self.name = name

    def __str__(self):
        """Representation of the scatterer."""
        sld = 1 - complex(self)  # Returns optical constant
        return f"n = {sld}"

    def __complex__(self):
        """Complex representation of the scatterer."""
        raise NotImplementedError

    @property
    def parameters(self):
        """Parameters."""
        raise NotImplementedError

    def __call__(self, thick=0, rough=0) -> Structure:
        """
        Create a :class:`PXR_Slab`.

        Parameters
        ----------
        thick: refnx.analysis.Parameter or float
            Thickness of slab in Angstrom
        rough: refnx.analysis.Parameter or float
            Roughness of slab in Angstrom

        Returns
        -------
        slab : refnx.PXR_reflect.PXR_Slab
            The newly made Slab with a dielectric tensor.

        Example
        --------
        >>> n_xx = complex(-0.0035, 0.0004)  # [unitless] #Ordinary Axis
        >>> n_zz = complex(-0.0045, 0.0009)  # [unitless] #Extraordinary Axis
        >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name="material")  # molecule
        >>> # Crete a slab with 10 A in thickness and 3 A roughness
        >>> slab = molecule(10, 3)

        """
        slab = Slab(thick, self, rough, name=self.name)
        slab.thick.setp(vary=True, bounds=(0, 2 * thick))
        slab.rough.setp(vary=True, bounds=(0, 2 * rough))
        return slab

    def __or__(self, other) -> Structure:
        """Combine scatterers."""
        # c = self | other
        slab = self()
        return slab | other


class SLD(Scatterer):
    """
    Object representing freely varying complex tensor index of refraction of a material.

    Parameters
    ----------
    value : float, complex, 'np.array'
        Valid np.ndarray.shape: (2,), (3,), (3,3) ('xx', 'yy', 'zz')
        tensor index of refraction.
        Units (N/A)
    symmetry : ('iso', 'uni', 'bi')
        Tensor symmetry. Automatically applies inter-parameter constraints.
    name : str, optional
        Name of object for later reference.

    Notes
    -----
    Components correspond to individual tensor components defined as ('xx', 'yy', 'zz').
    In a uniaxial approximation the following inputs are equivalent.

    >>> n_xx = complex(-0.0035, 0.0004)  # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009)  # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name="molecule")
    >>> molecule = PXR_SLD(np.array([n_xx, n_xx, n_zz], name='molecule')
    >>> molecule = PXR_SLD(np.array([n_xx, n_xx, n_zz])*np.eye(3), name='molecule)

    An PXR_SLD object can be used to create a PXR_Slab:

    >>> n_xx = complex(-0.0035, 0.0004)  # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009)  # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name="material")  # molecule
    >>> # Crete a slab with 10 A in thickness and 3 A roughness
    >>> slab = molecule(10, 3)

    Tensor symmetry can be applied using `symmetry`.

    >>> #'uni' will constrain n_xx = n_yy.
    >>> self.yy.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)

    >>> #'iso' will constrain n_xx = n_yy = n_zz
    >>> self.yy.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
    >>> self.zz.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.izz.setp(self.ixx, vary=None, constraint=self.ixx)
    """

    def __init__(
        self,
        value: np.ndarray | pd.DataFrame | float | complex | None = None,
        symmetry="uni",
        name="",
        energy=None,
    ):
        """Initialize SLD object.

        Parameters
        ----------
        value : array-like, DataFrame or scalar
            Can be:
            - DataFrame with columns ['energy', 'n_xx', 'n_ixx', 'n_zz', 'n_izz']
            - ndarray of shape (2,) or (3,) or (3,3) containing complex indices
            - scalar value for isotropic index
        symmetry : str
            'iso', 'uni' or 'bi' for symmetry constraints
        name : str
            Name of the SLD
        energy : float, optional
            Energy in eV to evaluate optical constants if value is DataFrame
        """
        super().__init__(name=name)

        # If value is DataFrame, interpolate at specified energy
        if isinstance(value, pd.DataFrame):
            if not energy:
                e = "Must specify energy when passing DataFrame"
                raise ValueError(e)

            required_cols = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
            if not all(col in value.columns for col in required_cols):
                e = f"DataFrame must contain columns: {required_cols}"
                raise ValueError(e)

            en_col = value["energy"].to_numpy()
            n_xx = value["n_xx"].to_numpy()
            n_zz = value["n_zz"].to_numpy()
            n_ixx = value["n_ixx"].to_numpy()
            n_izz = value["n_izz"].to_numpy()

            n_xx = np.interp(energy, en_col, n_xx)
            n_zz = np.interp(energy, en_col, n_zz)
            n_ixx = np.interp(energy, en_col, n_ixx)
            n_izz = np.interp(energy, en_col, n_izz)

            value = np.array([n_xx + 1j * n_ixx, n_xx + 1j * n_ixx, n_zz + 1j * n_izz])

        # Handle ndarray input
        elif isinstance(value, np.ndarray):
            if value.shape == (3,):
                pass
            elif value.shape == (2,):
                temp_val = np.ones(3) * value[0]
                temp_val[2] = value[1]
                value = temp_val
            elif value.shape == (3, 3):
                value = value.diagonal()
            else:
                e = "Array must have shape (2,), (3,) or (3,3)"
                raise ValueError(e)

        # Handle scalar input
        elif isinstance(value, (int, float, complex)):
            value = value * np.ones(3)
        else:
            e = "Input must be DataFrame, array or scalar"
            raise TypeError(e)

        # Create parameters
        self._parameters = Parameters(name=name)
        self.delta = Parameter(np.average(value).real, name=f"{name}_diso")
        self.beta = Parameter(np.average(value).imag, name=f"{name}_biso")

        self.xx = Parameter(value[0].real, name=f"{name}_{tensor_index[0]}")
        self.ixx = Parameter(value[0].imag, name=f"{name}_i{tensor_index[0]}")
        self.yy = Parameter(value[1].real, name=f"{name}_{tensor_index[1]}")
        self.iyy = Parameter(value[1].imag, name=f"{name}_i{tensor_index[1]}")
        self.zz = Parameter(value[2].real, name=f"{name}_{tensor_index[2]}")
        self.izz = Parameter(value[2].imag, name=f"{name}_i{tensor_index[2]}")

        self.birefringence = Parameter(
            (self.xx.value - self.zz.value), name=f"{name}_bire"
        )
        self.dichroism = Parameter(
            (self.ixx.value - self.izz.value), name=f"{name}_dichro"
        )

        self._parameters.extend(
            [
                self.delta,
                self.beta,
                self.xx,
                self.ixx,
                self.yy,
                self.iyy,
                self.zz,
                self.izz,
                self.birefringence,
                self.dichroism,
            ]
        )

        self.symmetry = symmetry

    def __repr__(self):
        """Representation of the scatterer."""
        return (
            "Isotropic Index of Refraction = ([{delta!r}, {beta!r}],"
            " name={name!r})".format(**self.__dict__)
        )

    def __complex__(self):
        """Complex representation of the scatterer."""
        sldc = complex(self.delta.value, self.beta.value)
        return sldc

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters`.

        associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    @property
    def symmetry(self):
        """
        Specify `symmetry` to automatically constrain the components.

        Default is 'uni'
        """
        return self._symmetry

    @symmetry.setter
    def symmetry(self, symmetry):
        self._symmetry = symmetry
        if self._symmetry == "iso":
            self.yy.setp(self.xx, vary=None, constraint=self.xx)
            self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
            self.zz.setp(self.xx, vary=None, constraint=self.xx)
            self.izz.setp(self.ixx, vary=None, constraint=self.ixx)
        elif self._symmetry == "uni":
            self.yy.setp(self.xx, vary=None, constraint=self.xx)
            self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
        elif self._symmetry == "bi":
            self.xx.setp(self.xx, vary=None, constraint=None)
            self.ixx.setp(self.ixx, vary=None, constraint=None)
            self.yy.setp(self.yy, vary=None, constraint=None)
            self.iyy.setp(self.iyy, vary=None, constraint=None)
            self.zz.setp(self.zz, vary=None, constraint=None)
            self.izz.setp(self.izz, vary=None, constraint=None)

    @property
    def tensor(self):  #
        """
        A full 3x3 matrix composed of the individual parameter values.

        Returns
        -------
            out : np.ndarray (3x3)
                complex tensor index of refraction
        """
        self._tensor = np.array(
            [
                [self.xx.value + 1j * self.ixx.value, 0, 0],
                [0, self.yy.value + 1j * self.iyy.value, 0],
                [0, 0, self.zz.value + 1j * self.izz.value],
            ],
            dtype=complex,
        )
        return self._tensor


class MaterialSLD(Scatterer):
    """
    Object representing complex index of refraction of a chemical formula.

    Only works for an isotropic material, convenient for substrate and superstrate
    materials
    You can fit the mass density of the material.
    Takes advantage of the PeriodicTable python package for calculations

    Parameters
    ----------
    formula : str
        Chemical formula
    density : float or Parameter
        mass density of compound in g / cm**3
    energy : float, optional
        energy of radiation (ev) ~ Converted to Angstrom in function
    name : str, optional
        Name of material

    Notes
    -----
    You need to have the `periodictable` package installed to use this object.
    A PXR_MaterialSLD object can be used to create a PXR_Slab:

    ```python
    # A PXR_MaterialSLD object for a common substrate
    en = 284.4  # [eV] Evaluate PeriodicTable at this energy
    sio2 = PXR_MaterialSLD("SiO2", density=2.4, energy=en, name="SiO2")  # Substrate
    si = PXR_MaterialSLD("Si", density=2.33, energy=en, name="SiO2")  # Substrate
    ```

    """

    def __init__(self, formula, density=None, energy=250.0, name=""):
        super().__init__(name=name)

        self.__formula = pt.formula(
            formula
        )  # Build the PeriodicTable object for storage
        self._compound = formula  # Keep a reference of the str object
        if density is None:
            density = compound_density(formula)
        self.density = possibly_create_parameter(
            density, name=f"{name}_rho", vary=True, bounds=(0, 5 * density)
        )

        self._energy = energy  # Store in eV for user interface
        self._wavelength = (
            hc / self._energy
        )  # Convert to Angstroms for later calculations
        self._tensor = None  # Build this when its called based in parameter values

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density])

    def __repr__(self):
        """Representation of the scatterer."""
        d = {
            "compound": self._compound,
            "density": self.density,
            "energy": self.energy,
            "wavelength": self.wavelength,
            "name": self.name,
        }
        return (
            "MaterialSLD({compound!r}, {density!r},"
            "energy={energy!r}, wavelength={wavelength!r}, name={name!r})".format(**d)
        )

    @property
    def formula(self):
        """
        Chemical formula used to calculate the index of refraction.

        Returns
        -------
            formula : str
                Full chemical formula used to calculate index of refraction.

        """
        return self._compound

    @formula.setter
    def formula(self, formula):
        import periodictable as pt

        self.__formula = pt.formula(formula)
        self._compound = formula

    @property
    def energy(self):
        """
        Photon energy to evaluate index of refraction in eV.

        Automatically updates wavelength when assigned.

        Returns
        -------
            energy : float
                Photon energy of X-ray probe.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy
        self._wavelength = (
            hc / self._energy
        )  # Update the wavelength if the energy changes

    @property
    def wavelength(self):
        """
        Wavelength to evaluate index of refraction in Angstroms.

        Automatically updates
        energy when assigned.

        Returns
        -------
            wavelength : float
                Wavelength of X-ray probe.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength
        self._energy = (
            hc / self._wavelength
        )  # Update the energy if the wavelength changes

    def __complex__(self):
        """Complex representation of the scatterer."""
        sldc = xsf.index_of_refraction(
            self.__formula, density=self.density.value, wavelength=self.wavelength
        )
        if (
            type(sldc).__module__ == np.__name__
        ):  # check if the type is accidentally cast into numpy.
            sldc = sldc.item()
        return 1 - sldc  # pt.xsf makes the type numpy affiliated...
        # __complex__ does not play nice so we reconvert with .item()

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters`.

           associated with this component

        """
        return self._parameters

    @property
    def tensor(self):
        """
        An isotropic 3x3 tensor composed of `complex(self.delta, self.beta)`.

        Returns
        -------
            tensor : np.ndarray
                complex tensor index of refraction
        """
        self._tensor = np.eye(3) * complex(self)
        return self._tensor


class UniTensorSLD(Scatterer):
    """
    Object representing uniaxial index of refraction based on experimental oocs.

    This object is useful for materials wthat have a uniaxial index of refraction
    tensor that can be constructed from experiments. The polar axis rotation and
    mass density can be fit as parameters.

    Parameters
    ----------
    ooc: pd.DataFrame
        Optical constants stored in a pandas DataFrame. Must have columns:
        - 'energy': Photon energy in eV (float)
        - 'n_xx':  δxx
        - 'n_ixx': βixx
        - 'n_zz':  δzz
        - 'n_izz': βzz
    rotation: float or Parameter
        Rotation of the tensor index of refraction in radians
    density: float or Parameter
        Mass density scaling factor (g/cm^3)
    energy : float
        Initial energy for optical constant lookup (eV)
    energy_offset : float or Parameter
        Energy offset for optical constant lookup (eV)
    name : str
        Name of material

    Examples
    --------
    ```python
    # Load optical constants from a file
    ooc = pd.read_csv("ooc.csv")
    # Create a uniaxial scatterer
    interface = UniTensorSLD(
        ooc, density=1.45, rotation=2 * np.pi, energy=250.0, name="ZnPc"
    )
    bulk = UniTensorSLD(ooc, density=1.45, rotation=0.0, energy=250.0, name="ZnPc")
    ```

    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        rotation=0,
        density=1.0,
        energy=250.0,
        energy_offset=0,
        name="",
    ):
        # =================/ Input Validation /================
        required_columns = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
        if not all(col in ooc.columns for col in required_columns):
            missing = [col for col in required_columns if col not in ooc.columns]
            e = f"Optical constants dataframe missing required columns: {missing}"
            raise ValueError(e)

        # =================/ Initialize /================
        self._parameters = Parameters(name=name)
        super().__init__(name=name)

        # ============/ Isotropic Parameters /===========
        self.density = possibly_create_parameter(
            density, name=f"{name}_density", bounds=(0, 5 * density), vary=True
        )
        self.rotation = possibly_create_parameter(
            rotation, name=f"{name}_rotation", vary=True, bounds=(-np.pi, np.pi)
        )
        # ============/ Optical Constants /===========
        self.energy = energy
        self._energy_offset = possibly_create_parameter(
            energy_offset, name=f"{name}_enOffset", vary=True, bounds=(-0.01, 0.01)
        )

        self.n_xx = interp1d(ooc["energy"], ooc["n_xx"])
        self.n_ixx = interp1d(ooc["energy"], ooc["n_ixx"])
        self.n_zz = interp1d(ooc["energy"], ooc["n_zz"])
        self.n_izz = interp1d(ooc["energy"], ooc["n_izz"])

        # Add parameters to parameter set
        self._parameters.extend([self.density, self.rotation, self.energy_offset])

    def __complex__(self):
        """Complex representation of the scatterer."""
        sldc = (2 * self.tensor[0, 0] + self.tensor[1, 1]) / 3
        return sldc

    def __repr__(self):
        """Representation of the scatterer."""
        return "Index of Refraction = ({n!r}, name={name!r})".format(**self.__dict__)

    @property
    def energy_offset(self):
        """
        Energy offset for optical constant lookup.

        Returns
        -------
            energy_offset : float
                Energy offset for optical constant lookup.
        """
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, energy_offset):
        self._energy_offset.value = energy_offset

    @property
    def n(self):
        """
        Optical constants of the material.

        Returns
        -------
            n : np.ndarray
                Optical constants of the material.
        """
        e = self.energy + self.energy_offset.value
        return np.diag(
            [
                self.n_xx(e) + self.n_ixx(e) * 1j,
                self.n_zz(e) + self.n_izz(e) * 1j,
            ]
        )

    @property
    def parameters(self):
        """
        Output the parameters associated with this component.
        """
        self._parameters.name = self.name
        return self._parameters

    @property
    def tensor(self):  #
        """
        A full 3x3 matrix composed of the individual parameter values.

        Returns
        -------
            out : np.ndarray (3x3)
                complex tensor index of refraction
        """
        n = self.density.value * self.n
        cos_squared = np.square(np.cos(self.rotation.value))
        sin_squared = 1 - cos_squared

        n_o = (n[0, 0] * (1 + cos_squared) + n[1, 1] * sin_squared) / 2
        n_e = n[0, 0] * sin_squared + n[1, 1] * cos_squared

        self._tensor = np.array(
            [
                [n_o, 0, 0],
                [0, n_o, 0],
                [0, 0, n_e],
            ],
            dtype=complex,
        )
        return self._tensor


class PXR_Component:
    """
    A base class for describing the structure of a subset of an interface.

    Parameters
    ----------
    name : str, optional
        The name associated with the Component

    Notes
    -----
    Currently limited to Gaussian interfaces.
    """

    def __init__(self, name=""):
        self.name = name

    def __or__(self, other):
        """
        OR'ing components can create a :class:`Structure`.

        Parameters
        ----------
        other: refnx.reflect.Structure, refnx.reflect.Component
            Combines with this component to make a Structure

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure

        Examples
        --------
        >>> air = SLD(0, name="air")
        >>> sio2 = SLD(3.47, name="SiO2")
        >>> si = SLD(2.07, name="Si")
        >>> structure = air | sio2(20, 3) | si(0, 3)

        """
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    def __mul__(self, n):
        """
        MUL'ing components makes them repeat.

        Parameters
        ----------
        n: int
            How many times you want to repeat the Component

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure
        """
        # convert to integer, should raise an error if there's a problem
        n = operator.index(n)
        if n < 1:
            return Structure()
        elif n == 1:
            return self
        else:
            s = Structure()
            s.extend([self] * n)
            return s

    def __str__(self):
        """Representation of the component."""
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters`.

         associated with this component
        """
        e = "A component should override the parameters property"
        raise NotImplementedError(e)

    def slabs(self, structure=None):
        """
        Slab representation of this component.

        Parameters
        ----------
        structure : PyPXR.anisotropic_reflect.PXR_Structure
            Summary of the structure that houses the component.

        Returns
        -------
        slabs : np.ndarray
            Slab representation of this Component.
            Has shape (N, 5).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               SLD.real of layer N (not including solvent)
            - slab[N, 2]
               *overall* SLD.imag of layer N (not including solvent)
            - slab[N, 3]
               roughness between layer N and N-1
            - slab[N, 4]
               volume fraction of solvent in layer N.

        If a Component returns None, then it doesn't have any slabs.
        """
        e = "A component should override the slabs property"
        raise NotImplementedError(e)

    def logp(self):
        """
        Log-probability for the component.

        Do not include log-probability terms for the actual parameters,
        these are automatically included elsewhere.

        Returns
        -------
        logp : float
            Log-probability
        """
        return 0


class Slab(PXR_Component):
    """
    A slab component has with tensor index of refraction associated over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld : :class:`PyPXR.anisotropic_structure.PXR_Scatterer`, complex, or float
        (complex) tensor index of refraction of film
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    """

    def __init__(self, thick, sld, rough, name=""):
        super().__init__(name=name)
        self.thick = possibly_create_parameter(thick, name=f"{name}_thick")
        if isinstance(sld, Scatterer):
            self.sld = sld
        else:
            self.sld = SLD(sld)

        self.rough = possibly_create_parameter(rough, name=f"{name}_rough")

        p = Parameters(name=self.name)
        p.extend([Parameters([self.thick, self.rough], name=f"{name}_slab")])
        p.extend(self.sld.parameters)

        self._parameters = p

    def __repr__(self):
        """Representation of the slab."""
        return f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r}, name={self.name!r},"

    def __str__(self):
        """Representation of the slab."""
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters`.

           associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component.

        See :class:`Component.slabs`
        """
        sldc = complex(self.sld)
        return np.array([[self.thick.value, sldc.real, sldc.imag, self.rough.value]])

    def tensor(self, energy=None):
        """
        Information pertaining to the tensor dielectric properties of the slab.

        Parameters
        ----------
        energy : float
            Updates PXR_SLD energy component associated with slab. Only required for
            PXR_MaterialSLD objects

        Returns
        -------
        tensor : np.ndarray
            Complex tensor index of refraction associated with slab.
        """
        if energy is not None and hasattr(self.sld, "energy"):
            self.sld.energy = energy
        return np.array([self.sld.tensor])


class MixedMaterialSlab(PXR_Component):
    """
    A slab component made of several components.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {anisotropic_reflect.PXR_Scatterer, complex, float}
        Sequence of materials that are contained in the slab.
    vf_list : sequence of refnx.analysis.Parameter or float
        relative volume fractions of each of the materials contained in the
        film.
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab

    Notes
    -----
    The index of refraction for this slab is calculated using the normalised volume
    fractions of each of the constituent components:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    """

    def __init__(
        self,
        thick,
        sld_list,
        vf_list,
        rough,
        name="",
    ):
        super().__init__(name=name)

        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        self.sld = []
        self.vf = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        for i, (s, v) in enumerate(zip(sld_list, vf_list, strict=False)):
            if isinstance(s, Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(SLD(s))

            self._sld_parameters.append(self.sld[-1].parameters)

            vf = possibly_create_parameter(v, name=f"vf{i} - {name}", bounds=(0.0, 1.0))
            self.vf.append(vf)
            self._vf_parameters.append(vf)

        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")

        p = Parameters(name=self.name)
        p.append(self.thick)
        p.extend(self._sld_parameters)
        p.extend(self._vf_parameters)
        p.extend([self.rough])

        self._parameters = p

    def __repr__(self):
        """Representation of the slab."""
        return (
            f"PXR_MixedMaterialSlab({self.thick!r}, {self.sld!r}, {self.vf!r},"
            f" {self.rough!r}, name={self.name!r},"
        )

    def __str__(self):
        """Representation of the slab."""
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters`.

           associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component.

        See :class:`Component.slabs`
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)

        sldc = np.sum(
            [
                complex(sld) * vf / sum_vfs
                for sld, vf in zip(self.sld, vfs, strict=False)
            ]
        )

        return np.array(
            [
                [
                    self.thick.value,
                    sldc.real,
                    sldc.imag,
                    self.rough.value,
                ]
            ]
        )

    def tensor(self, energy=None):
        """
        Information pertaining to the tensor dielectric properties of the slab.

        Parameters
        ----------
        energy : float
            Updates PXR_SLD energy component associated with slab. Only required for
            PXR_MaterialSLD objects

        Returns
        -------
        tensor : np.ndarray
            Complex tensor index of refraction associated with slab.
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)

        if energy is not None and hasattr(self.sld, "energy"):
            self.sld.energy = energy

        combinetensor = np.sum(
            [sld.tensor * vf / sum_vfs for sld, vf in zip(self.sld, vfs, strict=False)],
            axis=0,
        )

        return combinetensor  # self.sld.tensor


class Stack(PXR_Component, UserList):
    r"""
    Stack of PXR_Components.

    A series of PXR_Components that are considered as a single item. When
    incorporated into a PXR_Structure the PXR_Stack will be repeated as a multilayer

    Parameters
    ----------
    components : sequence
        A series of PXR_Components to repeat in a structure
    name: str
        Human readable name for the stack
    repeats: number, Parameter
        Number of times to repeat the stack within a structure to make a multilayer

    """

    def __init__(self, components=(), name="", repeats=1):
        PXR_Component.__init__(self, name=name)
        UserList.__init__(self)

        self.repeats = possibly_create_parameter(repeats, "repeat")
        self.repeats.bounds.lb = 1

        # Construct the list of components
        for c in components:
            if isinstance(c, PXR_Component):
                self.data.append(c)
            else:
                e = "You can only initialise a PXR_Stack with PXR_Components"
                raise TypeError(e)

    def __setitem__(self, i, v):
        """Set the ith item of the stack."""
        self.data[i] = v

    def __str__(self):
        """Representation of the stack."""
        s = []
        s.append("{:=>80}".format(""))

        s.append(f"Stack start: {round(abs(self.repeats.value))} repeats")
        for component in self:
            s.append(str(component))
        s.append("Stack finish")
        s.append("{:=>80}".format(""))

        return "/n".join(s)

    def __repr__(self):
        """Representation of the stack."""
        return "Stack(name={name!r}, components={data!r}, repeats={repeats!r}".format(
            **self.__dict__
        )

    def append(self, item):
        """
        Append a PXR_Component to the Stack.

        Parameters
        ----------
        item: PXR_Compponent
            PXR_Component to be added to the PXR_Stack

        """
        if isinstance(item, Scatterer):
            self.append(item())
            return

        if not isinstance(item, PXR_Component):
            e = "You can only add PXR_Components"
            raise TypeError(e)
        self.data.append(item)

    def slabs(self, structure=None):
        """
        Slab representation of this component.

        Notes
        -----
        Returns a list of each slab included within this Stack.

        """
        if not len(self):
            return None

        repeats = round(abs(self.repeats.value))

        slabs = np.concatenate([c.slabs(structure=self) for c in self.components])

        if repeats > 1:
            slabs = np.concatenate([slabs] * repeats)

        if hasattr(self, "solvent"):
            delattr(self, "solvent")

        return slabs

    def tensor(self, energy=None):
        """
        Tensor representation of this component.

        Builds list of all components
        """
        if not len(self):
            return None

        repeats = round(abs(self.repeats.value))

        tensor = np.concatenate(
            [c.tensor(energy=energy) for c in self.components], axis=0
        )

        if repeats > 1:
            tensor = np.concatenate([tensor] * repeats)

        return tensor

    @property
    def components(self):
        """
        List of components.
        """
        return self.data

    @property
    def parameters(self):
        r"""
        All Parameters associated with this Stack.

        """
        p = Parameters(name=f"Stack - {self.name}")
        p.append(self.repeats)
        p.extend([component.parameters for component in self.components])
        return p


def birefringence_profile(slabs, tensor, z=None, step=False):
    """
    Series of depth profiles for the slab model used to calculated p-RSoXR.

    Parameters
    ----------
    slabs : Information regarding the layer stack, see PXR_Structure class
    tensor : List of dielectric tensors from each layer stack, see PXR_Structure class
    z : float
        Interfacial distance (Angstrom) measured from interface between the
        fronting medium and the first layer.
    step : Boolean
        Set 'True' for slab model without interfacial widths


    Returns
    -------
    zed : float / np.ndarray
        Depth into the film / Angstrom

    index_tensor : complex / np.ndarray
        Real and imaginary tensor components of index of refraction / unitless
        Array elements: [nxx, nyy, nzz]

    Optional:

    index_step : complex / np.ndarray
        Real and imaginary tensor components of index of refraction / unitless
        Calculated WITHOUT interfacial roughness

    Notes
    -----
    This can be called in vectorised fashion.

    To calculate the isotropic components:
        index_iso = index_tensor.sum(axis=1)/3 #(nxx + nyy + nzz)/3
    To calculate the birefringence/dichroism:
        diff = index_tensor[:,0] - index_tensor[:,2] #nxx - nzz

    """
    nlayers = (
        np.size(slabs, 0) - 2
    )  # Calculate total number of layers (not including fronting/backing)

    # work on a copy of the input array
    layers = np.copy(slabs)
    layers[:, 0] = np.fabs(slabs[:, 0])  # Ensure the thickness is positive
    layers[:, 3] = np.fabs(slabs[:, 3])  # Ensure the roughness is positive
    # bounding layers should have zero thickness
    layers[0, 0] = layers[-1, 0] = 0

    # distance of each interface from the fronting interface
    dist = np.cumsum(layers[:-1, 0])
    total_film_thickness = int(
        np.round(dist[-1])
    )  # Total film thickness for point density
    # workout how much space the SLD profile should encompass
    # (if z array not provided)
    if z is None:
        zstart = -5 - 4 * np.fabs(slabs[1, 3])
        zend = 5 + dist[-1] + 4 * layers[-1, 3]
        zed = np.linspace(
            zstart, zend, num=total_film_thickness * 2
        )  # 0.5 Angstrom resolution default
    else:
        zed = np.asfarray(z)

    # Reduce the dimensionality of the tensor for ease of use
    reduced_tensor = tensor.diagonal(
        0, 1, 2
    )  # 0 - no offset, 1 - first axis of the tensor, 2 - second axis of the tensor

    tensor_erf = (
        np.ones((len(zed), 3), dtype=float) * reduced_tensor[0]
    )  # Full wave of initial conditions
    tensor_step = np.copy(tensor_erf)  # Full wave without interfacial roughness
    delta_n = reduced_tensor[1:] - reduced_tensor[:-1]  # Change in n at each interface

    # use erf for roughness function, but step if the roughness is zero
    step_f = Step()  # Step function (see refnx documentation)
    erf_f = Erf()  # Error function (see refnx documentation)
    sigma = layers[1:, 3]  # Interfacial width parameter

    # accumulate the SLD of each step.
    for i in range(nlayers + 1):
        f = erf_f
        g = step_f
        if sigma[i] == 0:
            f = step_f
        tensor_erf += (
            delta_n[None, i, :] * f(zed, scale=sigma[i], loc=dist[i])[:, None]
        )  # Broadcast into a single item
        tensor_step += (
            delta_n[None, i, :] * g(zed, scale=0, loc=dist[i])[:, None]
        )  # Broadcast into a single item

    return zed, tensor_erf if step is False else tensor_step


# Taken from Kas's code.


def compound_density(compound, desperate_lookup=True):
    """Density of the compound in g/cm^3.

    Elemental densities are taken from periodictable, which gets
    the densities from "The ILL Neutron Data Booklet, Second Edition."
    For compound densities, the values from the henke database at
    http://henke.lbl.gov/cgi-bin/density.pl are used
    if available.
    If the compound density is not found for the given compound, None is returned,
    unless desperate_lookup is True,
    in which case the elemental density of the first element in the compound is
    returned.
    """
    for d in henke_densities:
        if compound in (d[0], d[1]):
            return d[2]
    comp = pt.formula(compound)
    if comp.density is not None:
        return comp.density
    if desperate_lookup:
        return comp.structure[0][1].density
    return None


henke_densities = [
    ["", "AgBr", 6.473],
    ["", "AlAs", 3.81],
    ["", "AlN", 3.26],
    ["Sapphire", "Al2O3", 3.97],
    ["", "AlP", 2.42],
    ["", "B4C", 2.52],
    ["", "BeO", 3.01],
    ["", "BN", 2.25],
    ["Polyimide", "C22H10N2O5", 1.43],
    ["Polypropylene", "C3H6", 0.90],
    ["PMMA", "C5H8O2", 1.19],
    ["Polycarbonate", "C16H14O3", 1.2],
    ["Kimfol", "C16H14O3", 1.2],
    ["Mylar", "C10H8O4", 1.4],
    ["Teflon", "C2F4", 2.2],
    ["Parylene-C", "C8H7Cl", 1.29],
    ["Parylene-N", "C8H8", 1.11],
    ["Fluorite", "CaF2", 3.18],
    ["", "CdWO4", 7.9],
    ["", "CdS", 4.826],
    ["", "CoSi2", 5.3],
    ["", "Cr2O3", 5.21],
    ["", "CsI", 4.51],
    ["", "CuI", 5.63],
    ["", "InN", 6.88],
    ["", "In2O3", 7.179],
    ["", "InSb", 5.775],
    ["", "IrO2", 11.66],
    ["", "GaAs", 5.316],
    ["", "GaN", 6.10],
    ["", "GaP", 4.13],
    ["", "HfO2", 9.68],
    ["", "LiF", 2.635],
    ["", "LiH", 0.783],
    ["", "LiOH", 1.43],
    ["", "MgF2", 3.18],
    ["", "MgO", 3.58],
    ["", "Mg2Si", 1.94],
    ["Mica", "KAl3Si3O12H2", 2.83],
    ["", "MnO", 5.44],
    ["", "MnO2", 5.03],
    ["", "MoO2", 6.47],
    ["", "MoO3", 4.69],
    ["", "MoSi2", 6.31],
    ["Salt", "NaCl", 2.165],
    ["", "NbSi2", 5.37],
    ["", "NbN", 8.47],
    ["", "NiO", 6.67],
    ["", "Ni2Si", 7.2],
    ["", "Ru2Si3", 6.96],
    ["", "RuO2", 6.97],
    ["", "SiC", 3.217],
    ["", "Si3N4", 3.44],
    ["Silica", "SiO2", 2.2],
    ["Quartz", "SiO2", 2.65],
    ["", "TaN", 16.3],
    ["", "TiN", 5.22],
    ["", "Ta2Si", 14.0],
    ["Rutile", "TiO2", 4.26],
    ["ULE", "Si.925Ti.075O2", 2.205],
    ["", "UO2", 10.96],
    ["", "VN", 6.13],
    ["Water", "H2O", 1.0],
    ["", "WC", 15.63],
    ["YAG", "Y3Al5O12", 4.55],
    ["Zerodur", "Si.56Al.5P.16Li.04Ti.02Zr.02Zn.03O2.46", 2.53],
    ["", "ZnO", 5.675],
    ["", "ZnS", 4.079],
    ["", "ZrN", 7.09],
    ["Zirconia", "ZrO2", 5.68],
    ["", "ZrSi2", 4.88],
]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from pyref.fitting import AnisotropyObjective
    from pyref.fitting.reflectivity import ReflectModel
    from pyref.fitting.refnx_converters import XrayReflectDataset

    sns.set_palette("blend:#00829c,#ff9d8d", n_colors=3)

    ooc = pd.read_csv("~/.projects/pyref/optical_constants.csv")
    si = MaterialSLD("Si", name="Si", energy=283.7)(0, 1.5)

    fig, ax = plt.subplots(
        1, 2, figsize=(8, 2), sharey=True, gridspec_kw={"wspace": 0.1}
    )

    znpc_slab = UniTensorSLD(
        ooc,
        rotation=0,
        density=1.45,
        energy=283.7,
        name="ZnPC",
    )(196.441, 7.216)
    vac = MaterialSLD("", density=None, name="Vac")(0, 0)

    struct = vac | znpc_slab | si
    struct.name = "ZnPC/Si"
    struct.plot(ax=ax[0])

    znpc_slab = UniTensorSLD(
        ooc,
        rotation=np.pi / 2,
        density=1.45,
        energy=283.7,
        name="ZnPC",
    )(196.441, 7.216)

    struct = vac | znpc_slab | si
    struct.plot(ax=ax[1])
    print(struct)
    plt.show()

    # Plot a model
    models = ReflectModel(struct, name="ZnPC/Si", energy=283.7, pol="s")
    modelp = ReflectModel(struct, name="ZnPC/Si", energy=283.7, pol="p")
    model = ReflectModel(struct, name="ZnPC/Si", energy=283.7, pol="sp")

    q = np.linspace(0.01, 0.1, 100)
    datas = models(q)
    datap = modelp(q)
    xraydata = XrayReflectDataset.from_arrays(
        x_s=q, y_s=datas, x_p=q, y_p=datap, name="ZnPC/Si"
    )
    obj = AnisotropyObjective(model, xraydata, name="ZnPC/Si")
    ax, ax_ani = obj.plot()
    plt.show()
