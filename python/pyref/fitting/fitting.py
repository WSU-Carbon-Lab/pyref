import numpy as np
from core.frame import OrientedOpticalConstants
from pypxr.structure import PXR_Scatterer
from refnx.analysis import Parameter, Parameters, possibly_create_parameter

tensor_index = ["xx", "yy", "zz"]

# TODO: Check this out if it works


class PXR_NexafsSLD(PXR_Scatterer):
    """
    Object representing freely varying complex tensor index of refraction of a material.

    Index of refraction is caclulated using the nexafs data.

    Parameters
    ----------
    optical_constant : OrientedOpticalConstants
        Function that returns the optical constants for a given energy and density.
    density : float
        Density of material [g/cm^3].
    energy : float
        Energy of the x-ray beam [eV].
    symmetry : ('iso', 'uni', 'bi')
        Tensor symmetry. Automatically applies inter-parameter constraints.
    name : str, optional
        Name of object for later reference.
    en_offset : float, optional
        Energy offset for the nexafs data, default is 0.0.

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
        optical_constant: OrientedOpticalConstants,
        density,
        energy,
        name=None,
        en_offset=0.0,
    ):
        super().__init__(name=name)
        self.imag = Parameter(0, name="%s_isld" % name)
        self._tensor = None
        self.density = possibly_create_parameter(density, name="%s_rho" % name)
        self._optical_constant = optical_constant

        self._parameters = Parameters(name=name)
        self.delta = Parameter(optical_constant.delta(energy), name="%s_delta" % name)
        self.beta = Parameter(optical_constant.beta(energy), name="%s_beta" % name)

        self.xx = Parameter(
            optical_constant.xx(energy), name=f"{name}_{tensor_index[0]}"
        )
        self.ixx = Parameter(
            optical_constant.ixx(energy), name=f"{name}_i{tensor_index[0]}"
        )
        self.yy = Parameter(
            optical_constant.yy(energy), name=f"{name}_{tensor_index[1]}"
        )
        self.iyy = Parameter(
            optical_constant.iyy(energy), name=f"{name}_i{tensor_index[1]}"
        )
        self.zz = Parameter(
            optical_constant.zz(energy), name=f"{name}_{tensor_index[2]}"
        )
        self.izz = Parameter(
            optical_constant.izz(energy), name=f"{name}_i{tensor_index[2]}"
        )

        self.birefringence = Parameter(
            (self.xx.value - self.zz.value), name="%s_bire" % name
        )  # Useful parameters to use as constraints
        self.dichroism = Parameter(
            (self.ixx.value - self.izz.value), name="%s_dichro" % name
        )  # Defined in terms of xx and zz

        self.en_offset = Parameter((en_offset), name="%s_enOffset" % name)

        self._parameters.extend(
            [
                self.delta,
                self.beta,
                self.en_offset,
                self.xx,
                self.ixx,
                self.yy,
                self.iyy,
                self.zz,
                self.izz,
                self.birefringence,
                self.dichroism,
                self.density,
            ]
        )

        self.symmetry = optical_constant.symmetry

    @property
    def symmetry(self):
        """Specify `symmetry` to automatically constrain the components. Default is 'uni'."""
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

    def __complex__(self):
        _, sldc = self._optical_constant(self.density, self.energy + self.en_offset)
        return sldc
