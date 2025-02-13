"""
Testing for using Pre-Existing Optical Constants in Refnx.

The idea is to use pre calculated optical constants to define at least the initial
guess of the fitting parameters, allong with constants like the dichroism, and
birefringence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pypxr.structure import PXR_MaterialSLD, PXR_Scatterer  # type: ignore
from refnx.analysis import Parameter, Parameters, possibly_create_parameter

if TYPE_CHECKING:
    import pandas as pd


class PXR_NexafsSLD(PXR_Scatterer):
    """
    Object representing complex index of refraction based on experimental oocs.

    Parameters
    ----------
    ooc: pd.DataFrame
        Optical constants stored in a pandas DataFrame. Must have columns:
        - 'energy': Photon energy in eV (float)
        - 'n_xx':  δxx
        - 'n_ixx': βixx
        - 'n_zz':  δzz
        - 'n_izz': βzz
    symmetry : ('iso', 'uni', 'bi')
        Tensor symmetry. Automatically applies inter-parameter constraints.
    rotation: float or Parameter
        Rotation of the tensor index of refraction in radians
    density: float or Parameter
        Mass density scaling factor (g/cm^3)
    optical_constraints: Literal['none', 'biref', 'dicrho', 'all'] = 'none'
        How to constrain the tensor during fitting:
        - 'none': No constraints, tensor free to vary with symmetry constraints
        - 'dicrho': Constrain to match sign of predicted dichroism
        - 'biref': Constrain to match sign of predicted birefringence
        - 'dicrho-lite': Constrain to match predicted dichroism's sign
        - 'biref-lite': Constrain to match predicted birefringence's sign
        - 'all': Constrain to match predicted optical model
    energy : float
        Initial energy for optical constant lookup (eV)
    name : str
        Name of material
    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        symmetry="uni",
        rotation=0,
        density=1.0,
        optical_constraints: Literal["none", "biref", "dicrho", "all"] = "none",
        energy=250.0,
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

        # =========/ Get the Optical Constants /=========
        super().__init__(name=name)

        # =========/ Core Parameters /=========
        self.density = possibly_create_parameter(
            density, name=f"{name}_density", bounds=(0, None)
        )
        self.rotation = possibly_create_parameter(rotation, name=f"{name}_rotation")

        # =================/ Get optical constants for initial energy /================
        # Get initial optical constants from dataframe
        if energy in ooc["energy"].values:
            n_xx = np.array(ooc.loc[ooc["energy"] == energy]["n_xx"])
            n_ixx = np.array(ooc.loc[ooc["energy"] == energy]["n_ixx"])
            n_zz = np.array(ooc.loc[ooc["energy"] == energy]["n_zz"])
            n_izz = np.array(ooc.loc[ooc["energy"] == energy]["n_izz"])

        else:
            # Interpolate to get optical constants at desired energy
            n_xx = np.interp(energy, ooc["energy"], ooc["n_xx"])
            n_ixx = np.interp(energy, ooc["energy"], ooc["n_ixx"])
            n_zz = np.interp(energy, ooc["energy"], ooc["n_zz"])
            n_izz = np.interp(energy, ooc["energy"], ooc["n_izz"])
        # Store initial optical constants
        self._initial_tensor = np.diag([n_xx + n_ixx * 1j, n_zz + n_izz * 1j])
        self._initial_birefr = n_xx - n_zz
        self._initial_dichro = n_ixx - n_izz
        n = self._initial_tensor

        # =========/ Create Parameters /=========
        self.xx = Parameter(n[0, 0].real, name=f"{name}_xx")
        self.ixx = Parameter(n[0, 0].imag, name=f"{name}_ixx")
        self.yy = Parameter(n[0, 0].real, name=f"{name}_yy")
        self.iyy = Parameter(n[0, 0].imag, name=f"{name}_iyy")
        self.zz = Parameter(n[1, 1].real, name=f"{name}_zz")
        self.izz = Parameter(n[1, 1].imag, name=f"{name}_izz")

        self.delta = Parameter((2 * n_xx + n_zz) / 3, name=f"{name}_diso")
        self.beta = Parameter((2 * n_ixx + n_izz) / 3, name=f"{name}_biso")

        # birefringence/dichroism parameters
        self.birefringence = Parameter(self._initial_birefr, name=f"{name}_biref")
        self.dichroism = Parameter(self._initial_dichro, name=f"{name}_dichro")

        # Add parameters to parameter set
        self._parameters.extend(
            [
                self.density,
                self.rotation,
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

        # Set symmetry and optical constraints
        self._optical_constraints = None
        self.symmetry = symmetry
        self.optical_constraints = optical_constraints

    def __complex__(self):
        """Complex representation of the scatterer."""
        sldc = complex(self.delta.value, self.beta.value)
        return sldc

    def __repr__(self):
        """Representation of the scatterer."""
        return (
            "Isotropic Index of Refraction = ([{delta!r}, {beta!r}],"
            " name={name!r})".format(**self.__dict__)
        )

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
    def optical_constraints(self):
        """Optical constraint mode."""
        return self._optical_constraints

    @optical_constraints.setter
    def optical_constraints(self, value):
        """Update optical constraints."""
        self._optical_constraints = value
        match value:
            case "none":
                # No constraints - tensor free to vary with symmetry constraints
                # Reset any existing constraints
                self.xx.setp(self.xx, constraint=None)
                self.ixx.setp(self.ixx, constraint=None)
                self.zz.setp(self.zz, constraint=None)
                self.izz.setp(self.izz, constraint=None)

            case "biref":
                # Constrain to match sign of predicted birefringence
                # Set constraints to match initial birefringence
                self.birefringence.constraint = self._initial_biref
                self.dichroism.constraint = None
                self.xx.setp(
                    self.zz, vary=None, constraint=(self.xx.value + self._initial_biref)
                )

            case "biref-lite":
                # Constrain to match sign of predicted birefringence
                # Set bounds
                self.birefringence.bounds = (
                    (-np.inf, 0) if self._initial_biref < 0 else (0, np.inf)
                )

            case "dichro":
                # Constrain to match sign of predicted dichroism
                # Set constraints to match initial dichroism
                self.birefringence.constraint = None
                self.dichroism.constraint = self._initial_dichro
                self.ixx.setp(
                    self.izz,
                    vary=None,
                    constraint=(self.ixx.value + self._initial_dichro),
                )

            case "dichro-lite":
                # Constrain to match sign of predicted dichroism
                # Set bounds
                self.dichroism.bounds = (
                    (-np.inf, 0) if self._initial_dichro < 0 else (0, np.inf)
                )

            case "all":
                # constrain to match the optical model after rotation and density
                # scaling
                R = np.array(
                    [
                        [np.cos(self.rotation.value), -np.sin(self.rotation.value)],
                        [np.sin(self.rotation.value), np.cos(self.rotation.value)],
                    ]
                )
                n = self.density * R @ self._initial_ooc
                self.xx.setp(self.xx, vary=None, constraint=n[0, 0].real)
                self.ixx.setp(self.ixx, vary=None, constraint=n[0, 0].imag)
                self.zz.setp(self.zz, vary=None, constraint=n[1, 1].real)
                self.izz.setp(self.izz, vary=None, constraint=n[1, 1].imag)

    @property
    def tensor(self):  #
        """
        A full 3x3 matrix composed of the individual parameter values.

        Returns
        -------
            out : np.ndarray (3x3)
                complex tensor index of refraction
        """
        R = np.array(
            [
                [np.cos(self.rotation.value), -np.sin(self.rotation.value)],
                [np.sin(self.rotation.value), np.cos(self.rotation.value)],
            ]
        )
        n_2d = np.array(
            [
                [self.xx.value + 1j * self.ixx.value, 0],
                [0, self.zz.value + 1j * self.izz.value],
            ],
            dtype=complex,
        )
        n = self.density.value * (R @ n_2d)
        self._tensor = np.array(
            [
                [n[0, 0], 0, 0],
                [0, n[0, 0], 0],
                [0, 0, n[1, 1]],
            ],
            dtype=complex,
        )
        return self._tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    ooc = pd.read_csv("/home/hduva/projects/pyref/optical_constants.csv")
    si = PXR_MaterialSLD("Si")(0, 1.5)
    znpc_slab = PXR_NexafsSLD(
        ooc,
        symmetry="uni",
        rotation=90,
        density=2,
        optical_constraints="none",
        energy=283.7,
    )(196.441, 7.216)
    vac = PXR_MaterialSLD("", density=None)(0, 0)

    struct = vac | znpc_slab | si
    struct.plot()
    plt.show()
