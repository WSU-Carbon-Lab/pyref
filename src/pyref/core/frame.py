"""
ReflDataFrame
-------------
A subclass of pandas.DataFrame that contains 2d Data, metadata, and 
associated methods for working with reflectometry data.

@Author: Harlan Heilman
"""

import json
import pickle
import warnings
from calendar import c
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kkcalc import data, kk
from matplotlib.pylab import f
from matplotlib.testing import set_reproducibility_for_testing
from periodictable.xsf import index_of_refraction
from scipy.interpolate import interp1d

from .io import *

ArrayLike = Union[np.ndarray, list]
PathLike = Union[Path, str, NexafsIO]

def _update_kwargs(kwargs: dict, new_kws: dict)-> dict:
    kwargs.update(new_kws)
    return kwargs


def init_db(path:str | Path) -> None:
    """
    Initialize a database for storing data.

    Parameters
    ----------
    path : str | Path
        The path to the database.
    """
    path = Path(path) / ".db"
    # Save the database location to the config file for easy access
    config = Path(__file__).parent / "config.json"
    config_json = {"db": str(path)}
    with open(config, "w") as f:
        json.dump(config_json, f, indent=4)

    path.mkdir(parents=True, exist_ok=True)
    (path / ".data").mkdir(parents=True, exist_ok=True)
    (path / ".ocs").mkdir(parents=True, exist_ok=True)
    (path / ".struct").mkdir(parents=True, exist_ok=True)

    dbjson = {
        ".data": {
            "nexafs": [],
            "xrr": [],
        },
        ".ocs": [],
        ".struct": [],
    }

    with open(path / "db.json", "w") as f:
        json.dump(dbjson, f, indent=4)


def kkcalc(energy: np.ndarray, nexafs: np.ndarray, density: float, molecular_name: str, anchor: np.ndarray | None = None, **kwargs):

    nexafs = np.column_stack((energy, nexafs))

    xmin = energy[0]
    xmax = energy[-1]

    kws = {
        "load_options": None,
        "input_data_type": "Beta",
        "merge_points": [xmin, xmax],
        "add_background": False,
        "fix_distortions": False,
        "curve_tolerance": 0.05,
        "curve_recursion": 100,
    }

    kws.update(kwargs)

    scattering_factor = kk.kk_calculate_real(nexafs, molecular_name, **kws)

    extended_energy = scattering_factor[:, 0]
    f = interp1d(extended_energy, scattering_factor[:, 1], kind='linear', fill_value='extrapolate') #type: ignore
    fp = interp1d(extended_energy, scattering_factor[:, 2], kind='linear', fill_value='extrapolate') #type: ignore

    # Scale the scattering factors to optical constants using bare atom absorbtion
    if anchor is None:
        beta = -index_of_refraction(molecular_name, density=density, energy=[xmin*1e-3, xmax*1e-3]).imag
    
    else:
        beta = anchor
    
    lb = beta[0] / fp(xmin)
    ub = beta[-1] / fp(xmax)
    beta = interp1d(extended_energy, (lb + ub) / 2 * scattering_factor[:, 2], kind = 'linear', fill_value='extrapolate') #type: ignore
    delta = interp1d(extended_energy, (lb + ub) / 2 * scattering_factor[:, 1], kind = 'linear', fill_value='extrapolate') #type: ignore

    return delta, beta

    
class OpticalConstant:
    """ 
    A class to hold the optical constants for a given energy range.
    """
    def __init__(self, delta, beta) -> None:
        self.delta = delta
        self.beta = beta

    def __repr__(self) -> str:
        return f"OpticalConstants({self.delta}, {self.beta})"
    
    



class AngleNexafs(pd.DataFrame):
    """
    A subclass of pandas.DataFrame that allows for visualization of
    angle dependent NEXAFS data. All methods are in place.

    Returns
    -------
    pd : _type_
        _description_
    """

    # ----------------------------------------------------------------
    # constructors

    def __init__(
        self,
        nexafs: PathLike,
        molecular_name: str,
        density: float,
        angles: list = ["20", "40", "55", "70", "90"],
        name: str | None = None,
        *args,
        **kwargs,
    ):
        io = NexafsIO(nexafs)
        df = io.get_nexafs(angles=angles)
        self.__dict__.update(df.__dict__)

        self.io = io
        self.angles = " ".join(angles)
        self.molecular_name = molecular_name
        self.density = density
        self._name = name
        self.__db = Path(json.load(open(Path(__file__).parent / "config.json"))["db"])

    def __repr__(self):
        rep = super().__repr__()
        rep += "\n"
        rep += f"\nMolecular Name: {self.molecular_name}"
        rep += f"\nPath: {self.io.__repr__()}"
        rep += f"\nDensity: {self.density}"
        rep += f"\nAngles: {self.angles}"
        return rep

    def __str__(self):
        return self.__repr__()
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value
        else:
            raise TypeError("The name must be a string.")

    # ----------------------------------------------------------------
    # methods
    def plot_ds(self, **kwargs):
        fig, ax = plt.subplots(
            nrows=2, sharex=True ,gridspec_kw={"height_ratios": [3, 1], "hspace": 0}
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
        angs = self.angles.split(" ")

        fig, ax = plt.subplots(
            nrows=1,
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        super().plot(
            y=angs, ax=ax, ylabel="Nexafs [a.u]", title=f"{self.molecular_name if self.name is None else self.name} Nexafs", **kwargs,
        )
        ax.legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
            **kwargs,
        )

    def plot_beta(self, mpl_kw: dict[str, Any] | None = None, **kwargs):
        angs_ = self.angles.split(" ")
        angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]
        if mpl_kw is None:
            mpl_kw = {}

        try:
            fig, ax = plt.subplots(
                nrows=3,
                sharex=True,
                gridspec_kw={"height_ratios": [2, 3, 1], "hspace": 0},
                **mpl_kw,
            )
            self.plot(ax=ax[0], y=angs, legend=False, **kwargs)
            self.plot(ax=ax[2], y="Diff", legend=False, **kwargs)
            self.plot(
                ax=ax[1],
                y=[r"$\beta_{zz}$", r"$\beta_{iso}$", r"$\beta_{xx}$"],
                **kwargs,
            )

            ax[0].set(ylabel="Nexafs [a.u.]", title=f"{self.molecular_name if self.name is None else self.name} Optical Constants")
            ax[0].legend(
                title="Angle [deg]",
                labels=[rf"${a}^\circ$" for a in angs],
                loc="upper right",
            )
            ax[1].set(ylabel="Optical Constants [a.u.]")
            ax[2].set(ylabel="Diff [a.u.]", xlabel="Energy [eV]")
        except:
            print("Generate the out of plane spectrum first using df.get_ooc()")
    
    def plot_delta_beta(self, en_range: tuple | None = None, energy_highlights: list|np.ndarray|None = None,**kwargs):
        try:
            fig, ax = plt.subplots(
                nrows=2,
                sharex=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0},
            )
            if en_range is None:
                self.plot(ax=ax[0], y=[r"$\delta_{xx}$", r"$\delta_{zz}$", r"$\delta_{iso}$"], **kwargs)
                self.plot(ax=ax[1], y=[r"$\beta_{xx}$", r"$\beta_{zz}$", r"$\beta_{iso}$"], **kwargs)
            
            else: 
                energies = np.linspace(en_range[0], en_range[1], 1000)
                ax[0].plot(energies, self.xx.delta(energies), label=r"$\delta_{xx}$")
                ax[0].plot(energies, self.zz.delta(energies), label=r"$\delta_{zz}$")
                ax[0].plot(energies, self.iso.delta(energies), label=r"$\delta_{iso}$")
                ax[1].plot(energies, self.xx.beta(energies), label=r"$\beta_{xx}$")
                ax[1].plot(energies, self.zz.beta(energies), label=r"$\beta_{zz}$")
                ax[1].plot(energies, self.iso.beta(energies), label=r"$\beta_{iso}$")
                ax[0].legend()
                ax[1].legend()

            if energy_highlights is not None:
                for energy in energy_highlights:
                    ax[0].axvline(energy, color='magenta', linestyle='--', alpha=0.5)
                    ax[1].axvline(energy, color='magenta', linestyle='--', alpha=0.5)

            ax[0].set(ylabel=r"$\delta$ [a,u,]", title=f"Optical Constants {self.molecular_name if self.name is None else self.name}")
            ax[1].set(ylabel=r"$\beta$ [a.u.]")
            ax[1].set(xlabel="Energy [eV]")
        except:
            print("Generate the optical constants first using df.get_optical_constants()")

    def get_diffspec(self, plot=True):
        """
        Generate the difference spectrum.

        Returns
        -------
        pd.DataFrame
            The difference spectrum.
        """
        self["Diff"] = self["55"] - self["20"]
        if plot:
            self.plot_ds()

    def get_ooc(self, plot=True):
        """
        Generate the out of plane spectrum.

        Returns
        -------
        pd.DataFrame
            The out of plane spectrum.
        """
        if "Diff" not in self.columns:
            self.get_diffspec(plot=False)

        iso_intensity = self["55"].max()
        diff_intensity = self["Diff"].max()
        self[r"$\beta_{zz}$"] = self["55"]
        self[r"$\beta_{xx}$"] = self["55"]

        zsf = 2 * iso_intensity / diff_intensity
        xsf = (iso_intensity - self["55"].iloc[0]) / diff_intensity

        self[r"$\beta_{zz}$"] += zsf * self["Diff"]
        self[r"$\beta_{xx}$"] -= xsf * self["Diff"]
        self[r"$\beta_{iso}$"] = self["55"]

        if plot:
            self.plot_beta()

    def get_bare_atom(self):
        """
        Calculate the bare atom scattering factors and add them to the
        dataframe.
        """
        n = index_of_refraction(
            self.molecular_name, density=self.density, energy=self.index.values * 1e-3
        )

        self[r"$\delta_{ba}$"] = 1 - n.real
        self[r"$\beta_{ba}$"] = -n.imag

    def normalize(self):
        """
        Uses the bare atom index of refraction to normalize the beta columns.
        """
        if r"$\beta_{ba}$" not in self.columns:
            self.get_bare_atom()

        lb = self[r"$\beta_{iso}$"].iloc[0] / self[r"$\beta_{ba}$"].iloc[0]
        ub = self[r"$\beta_{iso}$"].iloc[-1] / self[r"$\beta_{ba}$"].iloc[-1]
        self[[r"$\beta_{iso}$", r"$\beta_{xx}$", r"$\beta_{zz}$"]] /= (lb + ub) / 2
    
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
        delta_xx, beta_xx = kkcalc(self.index.values, self[r"$\beta_{xx}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)
        delta_zz, beta_zz = kkcalc(self.index.values, self[r"$\beta_{zz}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)
        delta_iso, beta_iso = kkcalc(self.index.values, self[r"$\beta_{iso}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)

        self.xx = OpticalConstant(delta_xx, beta_xx)
        self.zz = OpticalConstant(delta_zz, beta_zz)
        self.iso = OpticalConstant(delta_iso, beta_iso)

        self[r"$\delta_{xx}$"] = self.xx.delta(self.index.values)
        self[r"$\delta_{zz}$"] = self.zz.delta(self.index.values)
        self[r"$\delta_{iso}$"] = self.iso.delta(self.index.values)

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
    
    def to_db(self):
        """
        Save the dataframe to a "database". This creates 3 files:
        
        ---

        * .parquet - the entire dataframe is saved as a parquet file.
        * .nexafs - the nexafs data is saved as a csv file with the nexafs data.
        * .oc - the delta, beta, interpolated functions are pickled and saved as a .oc file.


        Parameters
        ----------
        path : str | Path
            The path to save the database to.
        """

        
        with open(self.__db / "db.json", "r") as f:
            db = json.load(f)

            if self.molecular_name in db[".data"]["nexafs"]:
                warnings.warn("The molecular name already exists in the database. The data will be overwritten.")
                db[".data"]["nexafs"].remove(f"{self.molecular_name}.parquet")
                db[".data"]["nexafs"].remove(f"{self.molecular_name}.nexafs")
                db[".ocs"].remove(f"{self.molecular_name}.oc")
            
            db[".data"]["nexafs"].append(f"{self.molecular_name}.parquet")
            db[".data"]["nexafs"].append(f"{self.molecular_name}.nexafs")
            db[".ocs"].append(f"{self.molecular_name}.oc")


        parquet = self.__db / ".data"/ "nexafs" / f"{self.molecular_name}.parquet"
        nexafs = self.__db / ".data"/ "nexafs" / f"{self.molecular_name}.nexafs"
        ocs = self.__db / ".ocs" / f"{self.molecular_name}.oc"
        
        
        self.to_parquet(parquet)
        self.to_csv(nexafs)

        optical_model = {
            "xx": self.xx,
            "zz": self.zz,
            "iso": self.iso,
        }

        pickle.dump(optical_model, open(ocs, "wb"))

        

class ReflDataFrame(pd.DataFrame):
    """
    A subclass of pandas.DataFrame that contains 2d Data, metadata, and
    associated methods for working with reflectometry data.

    Parameters
    ----------
    pd : _type_
        _description_
    """

    # --------------------------------------------------------------
    # constructors

    def __init__(self, raw_images: ReflIO | None = None, meta_data=None, *args, **kwargs):
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
            except:
                warnings.warn("The path provided is not a valid path.")

    def __repr__(self):
        return super().__repr__() + "\n" + self.meta_data.__repr__()

    # --------------------------------------------------------------
    # properties

    @property
    def raw_images(self):
        return self._raw_images

    @raw_images.setter
    def raw_images(self, value):
        self._raw_images = value

    @property
    def meta_data(self):
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
            return refl
        except KeyError:
            warnings.warn("The dataframe does not contain the required columns.")
            return None
