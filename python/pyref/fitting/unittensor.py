from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from scipy.interpolate import interp1d

from pyref.fitting.structure import Scatterer

if TYPE_CHECKING:
    from numpy.typing import NDArray

speed_of_light = 299792458  # m/s
plank_constant = 4.135667697e-15  # ev*s
hc = (speed_of_light * plank_constant) * 1e10  # ev*A

tensor_index = ["xx", "yy", "zz"]  # Indexing for later definitions


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
        rotation: float = 0,
        density: float = 1.0,
        energy: float = 250.0,
        energy_offset: float = 0,
        name: str = "",
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
        self.density: Parameter = possibly_create_parameter(  # type: ignore[assignment]
            density, name=f"{name}_density", bounds=(0, 5 * density), vary=True
        )
        self.rotation: Parameter = possibly_create_parameter(  # type: ignore[assignment]
            rotation, name=f"{name}_rotation", vary=True, bounds=(-np.pi, np.pi)
        )
        # ============/ Optical Constants /===========
        self.energy = energy
        self.energy_offset: Parameter = possibly_create_parameter(  # type: ignore[assignment]
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-0.01, 0.01)
        )
        # store the optical constants as n_xx n_ixx, n_zz, n_izz
        self.n_xx = interp1d(ooc["energy"], ooc["n_xx"], bounds_error=False)
        self.n_ixx = interp1d(ooc["energy"], ooc["n_ixx"], bounds_error=False)
        self.n_zz = interp1d(ooc["energy"], ooc["n_zz"], bounds_error=False)
        self.n_izz = interp1d(ooc["energy"], ooc["n_izz"], bounds_error=False)

        # Add parameters to parameter set
        self._parameters.extend([self.density, self.rotation, self.energy_offset])

    def __complex__(self):
        """Complex representation of the scatterer."""
        sldc = (2 * self.tensor[0, 0] + self.tensor[1, 1]) / 3
        return sldc

    def __repr__(self):
        """Representation of the scatterer."""
        return "Index of Refraction = (name={name!r})".format(**self.__dict__)

    @property
    def n(self) -> NDArray[np.complex128]:
        """
        Optical constants of the material.

        Returns
        -------
        n : np.ndarray
            Optical constants of the material.
        """
        e = self.get_energy()
        return np.array(
            [
                [self.n_xx(e) + self.n_ixx(e) * 1j, 0],
                [0, self.n_zz(e) + self.n_izz(e) * 1j],
            ],
            dtype=np.complex128,
        )

    @property
    def parameters(self):
        """
        Output the parameters associated with this component.
        """
        self._parameters.name = self.name
        return self._parameters

    @property
    def tensor(self) -> NDArray[np.complex128]:
        """
        A full 3x3 matrix composed of the individual parameter values.

        Returns
        -------
            out : np.ndarray (3x3)
                complex tensor index of refraction
        """
        n: np.ndarray = self.get_density() * self.n
        cos_squared: float = np.square(np.cos(self.get_rotation()))
        sin_squared: float = 1 - cos_squared

        n_o: complex = (n[0, 0] * (1 + cos_squared) + n[1, 1] * sin_squared) / 2
        n_e: complex = n[0, 0] * sin_squared + n[1, 1] * cos_squared

        self._tensor = np.array(
            [
                [n_o, 0, 0],
                [0, n_o, 0],
                [0, 0, n_e],
            ],
            dtype=np.complex128,
        )
        return self._tensor
