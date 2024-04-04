import json
import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from warnings import deprecated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kkcalc import kk
from periodictable.xsf import index_of_refraction
from scipy.interpolate import interp1d

from pyref.core.io import NexafsIO, ReflIO

ArrayLike = np.ndarray | list

PathLike = Path | str | NexafsIO


def _update_kwargs(kwargs: dict, new_kws: dict) -> dict:
    kwargs.update(new_kws)
    return kwargs


def kkcalc(
    energy: np.ndarray,
    nexafs: np.ndarray,
    density: float,
    molecular_name: str,
    anchor: np.ndarray | None = None,
    **kwargs,
):
    """
    Summary.

    Parameters
    ----------
    energy : np.ndarray
        _description_
    nexafs : np.ndarray
        _description_
    density : float
        _description_
    molecular_name : str
        _description_
    anchor : np.ndarray | None, optional
        _description_, by default None
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    _type_
        _description_
    """
    nexafs = np.column_stack((energy, nexafs))

    xmin = energy[0]
    xmax = energy[-1]

    kws = {
        "load_options": None,
        "input_data_type": "nexafs",
        "merge_points": [xmin, xmax],
        "add_background": False,
        "fix_distortions": False,
        "curve_tolerance": 0.05,
        "curve_recursion": 100,
    }

    kws.update(kwargs)

    scattering_factor = kk.kk_calculate_real(nexafs, molecular_name, **kws)

    extended_energy = scattering_factor[:, 0]
    fp = interp1d(
        extended_energy,
        scattering_factor[:, 2],
        kind="linear",
        fill_value="extrapolate",
    )  # type: ignore

    # Scale the scattering factors to optical constants using bare atom absorbtion
    if anchor is None:
        beta = -index_of_refraction(
            molecular_name, density=density, energy=[xmin * 1e-3, xmax * 1e-3]
        ).imag

    else:
        beta = anchor

    lb = beta[0] / fp(xmin)
    ub = beta[-1] / fp(xmax)
    beta = interp1d(
        extended_energy,
        (lb + ub) / 2 * scattering_factor[:, 2],
        kind="linear",
        fill_value="extrapolate",
    )  # type: ignore
    delta = interp1d(
        extended_energy,
        (lb + ub) / 2 * scattering_factor[:, 1],
        kind="linear",
        fill_value="extrapolate",
    )  # type: ignore

    return delta, beta


class OpticalConstant:
    """A class to hold the optical constants for a given energy range."""

    # ---------------------------------------------------------------------------------
    # properties
    # ---------------------------------------------------------------------------------

    @property
    def density(self):
        """Get the density of the OpticalConstant object."""
        return self._density

    @density.setter
    def density(self, density: float, scale_location: float = 250):
        n = index_of_refraction(
            self.molecular_name, density=density, energy=scale_location * 1e-3
        )

        self = self * n.imag / self._beta(scale_location)

    def __call__(self, energy: float | list[float]):
        """
        Calculate the complex refractive index for a given energy.

        Parameters
        ----------
        energy : float | list[float]
            The energy value or a list of energy values.

        Returns
        -------
        complex: The complex refractive index.
        """
        return 1 - self.delta(energy) - 1j * self.beta(energy)

    def __repr__(self) -> str:
        return f"OpticalConstants({self.delta}, {self.beta})"

    def __add__(self, other):
        assert isinstance(other, complex)
        return lambda x: self.delta(x) + other.real + 1j * self.beta(x) + other.imag

    def __sub__(self, other):
        assert isinstance(other, complex)
        return lambda x: self.delta(x) - other.real + 1j * self.beta(x) - other.imag

    def __mul__(self, other):
        assert isinstance(other, complex)
        return lambda x: self.delta(x) * other.real + 1j * self.beta(x) * other.imag

    def __truediv__(self, other):
        assert isinstance(other, complex)
        return lambda x: self.delta(x) / other.real + 1j * self.beta(x) / other.imag

    def n(self, energy):
        """Calculate the index of refraction."""
        return self.delta(energy) + 1j * self.beta(energy)


class OrientedOpticalConstants(OpticalConstant):
    """A class to hold the optical constants for a given energy range."""

    def __init__(
        self,
        value: np.ndarray[OpticalConstant],
        symmetry,
        molecular_name: str,
        density=None,
    ) -> None:
        self.symmetry = symmetry
        self.molecular_name = molecular_name
        self._density = density

        if symmetry == "iso":
            assert len(value) == 1
            self.xx = value[0].delta
            self.yy = value[0].delta
            self.zz = value[0].delta
            self.ixx = value[0].beta
            self.iyy = value[0].beta
            self.izz = value[0].beta
        elif symmetry == "uni":
            assert len(value) == 2
            self.xx = value[0].delta
            self.yy = value[0].delta
            self.zz = value[1].delta
            self.ixx = value[0].beta
            self.iyy = value[0].beta
            self.izz = value[1].beta
        elif symmetry == "bi":
            assert len(value) == 3
            self.xx = value[0].delta
            self.yy = value[1].delta
            self.zz = value[2].delta
            self.ixx = value[0].beta
            self.iyy = value[1].beta
            self.izz = value[2].beta

        else:
            e = "Symmetry must be 'iso', 'uni', or 'bi'."
            raise ValueError(e)
        self._value = value

    # ---------------------------------------------------------------------------------
    # properties
    # ---------------------------------------------------------------------------------

    @property
    def density(self):
        """Get the density of the OrientedOpticalConstants object."""
        return self._density

    @density.setter
    def density(self, density: float, scale_location: float = 250):
        self.xx.density = (density, scale_location)
        self.yy.density = (density, scale_location)
        self.zz.density = (density, scale_location)
        self._density = density

    @property
    def delta(self):
        """Calculate the average of the delta values."""
        return lambda x: (self.xx(x) + self.yy(x) + self.zz(x)) / 3

    @property
    def beta(self):
        """Calculate the average of the beta values."""
        return lambda x: (self.ixx(x) + self.iyy(x) + self.izz(x)) / 3

    @property
    def birefringence(self):
        """Calculate the birefringence."""
        return lambda x: (self.xx(x) - self.zz(x))

    @property
    def dichroism(self):
        """Calculate the dichroism."""
        return lambda x: (self.ixx(x) - self.izz(x))

    def __call__(self, energy: float | list[float], density: float | None = None):
        """Return the optical constants for a given energy."""
        if density != self.density and density is not None:
            self.density = density

        if isinstance(energy, Iterable):
            n = []
            for e in energy:
                xx = self.xx(e) + 1j * self.ixx(e)
                zz = self.zz(e) + 1j * self.izz(e)
                n.append(np.asanyarray([xx, zz], dtype=complex))
            return n
        else:
            xx = self.xx(energy) + 1j * self.ixx(energy)
            zz = self.zz(energy) + 1j * self.izz(energy)
            return np.asarray([xx, zz], dtype=complex)

    def __repr__(self) -> str:
        return f"OrientedOpticalConstants({self.xx}, {self.yy}, {self.zz}, {self.ixx}, {self.iyy}, {self.izz})"

    def __str__(self) -> str:
        return super().__str__()


class AngleNexafs(pd.DataFrame):
    """
    A subclass of pandas.DataFrame.

    Returns
    -------
    pd : _type_
        _description_

    """

    # ----------------------------------------------------------------
    # constructors

    def __init__(
        self,
        nexafs: Path,
        molecular_name: str,
        density: float,
        angles: list[str] = ["20", "40", "55", "70", "90"],  # noqa: B006
        name: str | None = None,
        read_kwargs: dict | None = None,
    ):
        """A class to hold the optical constants for a given energy range."""
        io = NexafsIO(nexafs)
        df = io.get_nexafs(angles=angles, **(read_kwargs or {}))
        self.__dict__.update(df.__dict__)

        self.io = io

        if isinstance(angles, list):
            self.angles = " ".join(angles)
        else:
            self.angles = "55"

        self.molecular_name = molecular_name
        self.density = density
        self._name = name
        __db = json.load((Path(__file__).parent / "config.json").open())["db"]
        for string in __db:
            if Path(string).exists():
                self.__db = Path(string)
                break

    def __repr__(self):
        """Return a string representation of the object."""
        rep = super().__repr__()
        rep += "\n"
        rep += f"\nMolecular Name: {self.molecular_name}"
        rep += f"\nPath: {self.io.__repr__()}"
        rep += f"\nDensity: {self.density}"
        rep += f"\nAngles: {self.angles}"
        return rep

    def __str__(self):
        """Return a string representation of the object."""
        return self.__repr__()

    @property
    def name(self):
        """Get the name property."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name property."""
        if isinstance(value, str):
            self._name = value
        else:
            error_message = "The name must be a string."
            raise TypeError(error_message)

    # ----------------------------------------------------------------
    # methods
    def plot_ds(self, **kwargs):
        """Plot the ds."""
        fig, ax = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0}
        )
        angs_ = self.angles.split(" ")
        angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]

        self.plot(ax=ax[0], y=angs, **kwargs)
        self.plot(ax=ax[1], y="Diff", legend=False, **kwargs)

        ax[0].set(
            ylabel="Nexafs [a.u.]",
            title=f"{self.molecular_name if self.name is None else self.name} Optical Constants",
        )
        ax[0].legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
        )
        ax[1].set(ylabel=r"$\beta_{max} - \beta_{min}$", xlabel="Energy [eV]")

    def plot_ar(self, **kwargs):
        """Plot the ar."""
        angs = self.angles.split(" ")

        fig, ax = plt.subplots(
            nrows=1,
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        super().plot(
            y=angs,
            ax=ax,
            ylabel="Nexafs [a.u]",
            title=f"{self.molecular_name if self.name is None else self.name} Nexafs",
            **kwargs,
        )
        ax.legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
            **kwargs,
        )

    def plot_beta(self, mpl_kw: dict[str, Any] | None = None, **kwargs):
        """Plot the beta."""
        angs_ = self.angles.split(" ")
        if len(angs_) == 1:
            angs = angs_
        else:
            angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]
        if mpl_kw is None:
            mpl_kw = {}

        if "Diff" not in self.columns:
            fig_kws = {"nrows": 2, "gridspec_kw": {"hspace": 0}}
            diff = False
            beta_cols = r"$\beta_{iso}$"
        else:
            fig_kws = {
                "nrows": 3,
                "gridspec_kw": {"height_ratios": [2, 3, 1], "hspace": 0},
            }
            diff = True
            beta_cols = [r"$\beta_{zz}$", r"$\beta_{iso}$", r"$\beta_{xx}$"]

        fig, ax = plt.subplots(**fig_kws)
        self.plot(ax=ax[0], y=angs, legend=False, **kwargs)

        if diff:
            self.plot(ax=ax[2], y="Diff", legend=False, **kwargs)
            ax[2].set(ylabel="Diff [a.u.]", xlabel="Energy [eV]")
        else:
            ax[1].set(xlabel="Energy [eV]")

        self.plot(
            ax=ax[1],
            y=beta_cols,
            **kwargs,
        )

        ax[0].set(
            ylabel="Nexafs [a.u.]",
            title=f"{self.molecular_name if self.name is None else self.name} Optical Constants",
        )
        ax[0].legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
            loc="upper right",
        )
        ax[1].set(ylabel=r"$\beta$ [a.u.]")

    def plot_delta_beta(
        self,
        en_range: tuple | None = None,
        energy_highlights: list | np.ndarray | None = None,
        **kwargs,
    ):
        """Plot the delta beta."""
        fig, ax = plt.subplots(
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0},
        )
        if "Diff" not in self.columns:
            delta = [r"$\delta_{iso}$"]
            beta = [r"$\beta_{iso}$"]
            diff = False
        else:
            delta = [r"$\delta_{xx}$", r"$\delta_{zz}$", r"$\delta_{iso}$"]
            beta = [r"$\beta_{xx}$", r"$\beta_{zz}$", r"$\beta_{iso}$"]
            diff = True

        if en_range is None:
            self.plot(ax=ax[0], y=delta, **kwargs)
            self.plot(ax=ax[1], y=beta, **kwargs)

        else:
            energies = np.linspace(en_range[0], en_range[1], 1000)
            if not diff:
                ax[0].plot(energies, self.sld.delta(energies), label=r"$\delta_{iso}$")
                ax[1].plot(energies, self.sld.beta(energies), label=r"$\beta_{iso}$")
            else:
                ax[0].plot(energies, self.sld.xx(energies), label=r"$\delta_{xx}$")
                ax[0].plot(energies, self.sld.zz(energies), label=r"$\delta_{zz}$")
                ax[0].plot(energies, self.sld.delta(energies), label=r"$\delta_{iso}$")
                ax[1].plot(energies, self.sld.ixx(energies), label=r"$\beta_{xx}$")
                ax[1].plot(energies, self.sld.izz(energies), label=r"$\beta_{zz}$")
                ax[1].plot(energies, self.sld.beta(energies), label=r"$\beta_{iso}$")
            ax[0].legend()
            ax[1].legend()

        if energy_highlights is not None:
            for energy in energy_highlights:
                ax[0].axvline(energy, color="magenta", linestyle="--", alpha=0.5)
                ax[1].axvline(energy, color="magenta", linestyle="--", alpha=0.5)

        ax[0].set(
            ylabel=r"$\delta$ [a,u,]",
            title=f"Optical Constants {self.molecular_name if self.name is None else self.name}",
        )
        ax[1].set(ylabel=r"$\beta$ [a.u.]")
        ax[1].set(xlabel="Energy [eV]")

    def get_diffspec(self, plot=True):
        """
        Generate the difference spectrum.

        Returns
        -------
        pd.DataFrame
            The difference spectrum.
        """
        if len(self.columns) == 1:
            warnings.warn(
                "The NEXAFS data does not contain multiple angles.", stacklevel=2
            )
            return None

        self["Diff"] = self["55"] - self["20"]
        if plot:
            self.plot_ds()

    def get_ooc(self, plot=True, normalize=True):
        """
        Generate the out of plane spectrum.

        Returns
        -------
        pd.DataFrame
            The out of plane spectrum.
        """
        if "Diff" not in self.columns:
            self[r"$\beta_{iso}$"] = self["55"]
        else:
            iso_intensity = self["55"].max()
            diff_intensity = self["Diff"].max()
            self[r"$\beta_{zz}$"] = self["55"]
            self[r"$\beta_{xx}$"] = self["55"]
            # βzz(π*) = 3*βiso
            # βxx(π*) = 3/2βiso
            zsf = 2 * iso_intensity / diff_intensity
            xsf = (iso_intensity - self["55"].iloc[0]) / diff_intensity
            self[r"$\beta_{zz}$"] += zsf * self["Diff"]
            self[r"$\beta_{xx}$"] -= xsf * self["Diff"]
            self[r"$\beta_{iso}$"] = self["55"]
        if normalize:
            self.normalize()
        if plot:
            self.plot_beta()

    def get_bare_atom(self):
        """Calculate the bare atom scattering factors."""
        n = index_of_refraction(
            self.molecular_name, density=self.density, energy=self.index.values * 1e-3
        )

        self[r"$\delta_{ba}$"] = 1 - n.real
        self[r"$\beta_{ba}$"] = -n.imag

    def normalize(self):
        """Uses the bare atom index of refraction to normalize the beta columns."""
        if r"$\beta_{ba}$" not in self.columns:
            self.get_bare_atom()

        lb = self[r"$\beta_{iso}$"].iloc[0] / self[r"$\beta_{ba}$"].iloc[0]
        ub = self[r"$\beta_{iso}$"].iloc[-1] / self[r"$\beta_{ba}$"].iloc[-1]

        if "Diff" not in self.columns:
            warnings.warn(
                "The NEXAFS data does not contain multiple angles.", stacklevel=2
            )
        if len(self.columns) == 1:
            warnings.warn(
                "The NEXAFS data does not contain multiple angles.", stacklevel=2
            )
            self[r"$\beta_{iso}$"] /= (lb + ub) / 2
        else:
            try:
                self[[r"$\beta_{iso}$", r"$\beta_{xx}$", r"$\beta_{zz}$"]] /= (
                    lb + ub
                ) / 2
            except Exception as e:
                warnings.warn(
                    f"{e}\n Normalizing only the isotropic data. NEXAFS is likely only one angle",
                    stacklevel=2,
                )
                self[r"$\beta_{iso}$"] /= (lb + ub) / 2

    def get_kk(self, *args, **kwargs):
        """
        Calculate the optical constants using the KK method.

        Parameters
        ----------
        *args : _type_
            _description_
        **kwargs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if r"$\beta_{xx}$" in self.columns:
            delta_xx, beta_xx = kkcalc(
                self.index.values,
                self[r"$\beta_{xx}$"].to_numpy(),
                self.density,
                self.molecular_name,
                *args,
                **kwargs,
            )
            xx = OpticalConstant(
                delta_xx, beta_xx, self.molecular_name, density=self.density
            )
            self[r"$\delta_{xx}$"] = delta_xx(self.index.values)

        if r"$\beta_{zz}$" in self.columns:
            delta_zz, beta_zz = kkcalc(
                self.index.values,
                self[r"$\beta_{zz}$"].to_numpy(),
                self.density,
                self.molecular_name,
                *args,
                **kwargs,
            )
            zz = OpticalConstant(
                delta_zz, beta_zz, self.molecular_name, density=self.density
            )
            self[r"$\delta_{zz}$"] = delta_zz(self.index.values)

        if r"$\beta_{iso}$" in self.columns:
            delta_iso, _ = kkcalc(
                self.index.values,
                self[r"$\beta_{iso}$"].to_numpy(),
                self.density,
                self.molecular_name,
                *args,
                **kwargs,
            )
            self[r"$\delta_{iso}$"] = delta_iso(self.index.values)

        sld = [xx, zz]
        self.sld = OrientedOpticalConstants(
            sld, "uni", self.molecular_name, density=self.density
        )

    def get_optical_constants(self):
        """
        Calculate the optical constants using the KK method.

        Returns
        -------
        _type_
            _description_
        """
        self.get_bare_atom()
        self.get_ooc(plot=False)
        self.normalize()
        self.get_kk()

    def to_json(self, path):
        """
        Save the dataframe to a .csv file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        """
        from datetime import datetime

        # ---------------------------------------------------
        # Three data parts
        # 1. Header Data
        # 2. NEXAFS Data
        # 3. Optical Constants
        # ---------------------------------------------------

        data = {}
        data["header"] = {
            "molecule": self.molecular_name,
            "commonName": "ZnPc",
            "density": self.density,
            "angles": self.angles.split(" "),
            "synchroton": "Australia",
            "beamline": "ANSTO",
            "beamlineID": "XSTO",
            "scanType": "TEY",
            "deposition": "PVD",
            "Vendor": "",
            "Substrate": "N-type Si",
            "processTs": datetime.now().isoformat(),
            "collectionTs": "",
        }
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    def to_csv(self, path):
        """
        Save the dataframe to a CSV file.

        Parameters
        ----------
        path : str | Path
            The path to save the CSV file to.
        """
        df_nexafs = self[self.angles.split(" ")]
        oc_ens = np.linspace(50, 30000, 100000)
        oc_df = pd.DataFrame(
            {
                "Energy": oc_ens,
                "Delta": self.sld.delta(oc_ens),
                "Beta": self.sld.beta(oc_ens),
                "xx": self.sld.xx(oc_ens),
                "yy": self.sld.yy(oc_ens),
                "zz": self.sld.zz(oc_ens),
                "ixx": self.sld.ixx(oc_ens),
                "iyy": self.sld.iyy(oc_ens),
                "izz": self.sld.izz(oc_ens),
            }
        )

        df_nexafs.to_csv(path.with_suffix(".nexafs"), index=False)
        oc_df.to_csv(path.with_suffix(".oc"), index=False)

    def to_db(self):
        """
        Save the dataframe to a "database".

        ---

        * .parquet - the entire dataframe is saved as a parquet file.
        * .nexafs - the nexafs data is saved as a csv file with the nexafs data.
        * .oc - the delta, beta, interpolated functions are pickled and saved as a .oc

        Parameters
        ----------
        path : str | Path
            The path to save the database to.
        """
        with (self.__db / "db.json").open("r+") as f:
            data = json.load(f)

            if self.molecular_name in data["data"]["nexafs"]:
                data["data"]["nexafs"].remove(f"{self.molecular_name}")
                data["ocs"].remove(f"{self.molecular_name}")

            data["data"]["nexafs"].append(f"{self.molecular_name}")
            data["ocs"].append(f"{self.molecular_name}")

            f.seek(0)
            json.dump(data, f, indent=4)

        parquet = self.__db / ".data" / "nexafs" / f"{self.molecular_name}.parquet"
        nexafs = self.__db / ".data" / "nexafs" / f"{self.molecular_name}.nexafs"
        ocs = self.__db / ".ocs" / f"{self.molecular_name}.oc"
        dat = self.__db / ".data" / f"{self.molecular_name}.json"

        self.to_parquet(parquet)
        self[self.angles.split(" ")].to_csv(nexafs)
        with ocs.open("wb") as f:
            pickle.dump(self.sld, f)
        self.to_json(dat)
        self.to_csv(dat)


@deprecated
class ReflDataFrame(pd.DataFrame):
    """
    A subclass of pandas.DataFrame that contains 2d Data, metadata.

    Parameters
    ----------
    pd : _type_
        _description_
    """

    # --------------------------------------------------------------
    # constructors

    def __init__(
        self, raw_images: ReflIO | None = None, meta_data=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if isinstance(raw_images, ArrayLike):
            self.raw_images = raw_images
            self.meta_data = meta_data

        elif isinstance(raw_images, ReflIO):
            self.raw_images = raw_images.get_image()
            self.meta_data = raw_images.get_header()

        elif isinstance(raw_images, str):
            try:
                raw_images = ReflIO(raw_images)
                self.raw_images = raw_images.get_image()
                self.meta_data = raw_images.get_header()
            except Exception:
                warnings.warn("The path provided is not a valid path.", stacklevel=2)

    def __repr__(self):
        return super().__repr__() + "\n" + self.meta_data.__repr__()

    # --------------------------------------------------------------
    # properties

    @property
    def raw_images(self):
        """Get the raw_images property."""
        return self._raw_images

    @raw_images.setter
    def raw_images(self, value):
        self._raw_images = value

    @property
    def meta_data(self):
        """Get the meta_data property."""
        return self._meta_data

    @meta_data.setter
    def meta_data(self, value):
        self._meta_data = value

    # --------------------------------------------------------------
    # methods

    def to_npy(self, path):
        """
        Save the raw images to a .npy file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        """
        np.save(path, self.raw_images)

    def to_parquet(self, path):
        """
        Save the dataframe to a .parquet file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        """
        self.to_parquet(path)

    def to_refnx_dataset(self):
        """
        Convert the dataframe to a refnx.DataSet.

        Returns
        -------
        refnx.DataSet
            The refnx.DataSet object.
        """
        try:
            refl = (self["Q"], self["R"], self["dR"])
        except KeyError:
            warnings.warn(
                "The dataframe does not contain the required columns.", stacklevel=2
            )
            refl = None
        else:
            return refl
