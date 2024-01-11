"""
ReflDataFrame
-------------
A subclass of pandas.DataFrame that contains 2d Data, metadata, and 
associated methods for working with reflectometry data.

@Author: Harlan Heilman
"""

import json
import pickle
import re
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


def kkcalc(energy: np.ndarray, nexafs: np.ndarray, density: float, molecular_name: str, anchor: np.ndarray | None = None, **kwargs):

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

    def n(self, energy):
        return self.delta(energy) + 1j * self.beta(energy)
    
    



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
        nexafs: Path,
        molecular_name: str,
        density: float,
        angles: list | None = ["20", "40", "55", "70", "90"],
        name: str | None = None,
        read_kwargs: dict | None = None,
    ):
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
        if len(angs_) == 1:
            angs = angs_
        else:
            angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]
        if mpl_kw is None:
            mpl_kw = {}

        if "Diff" not in self.columns:
            fig_kws = {
                "nrows" : 2,
                "gridspec_kw" : {"hspace": 0}
            }
            diff = False
            beta_cols = r"$\beta_{iso}$"
        else:
            fig_kws = {
                "nrows" : 3,
                "gridspec_kw" : {"height_ratios": [2, 3, 1], "hspace": 0}
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

        ax[0].set(ylabel="Nexafs [a.u.]", title=f"{self.molecular_name if self.name is None else self.name} Optical Constants")
        ax[0].legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
            loc="upper right",
        )
        ax[1].set(ylabel=r"$\beta$ [a.u.]")
    
    def plot_delta_beta(self, en_range: tuple | None = None, energy_highlights: list|np.ndarray|None = None,**kwargs):
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
                ax[0].plot(energies, self.iso.delta(energies), label=r"$\delta_{iso}$")
                ax[1].plot(energies, self.iso.beta(energies), label=r"$\beta_{iso}$")
            else:
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

    def get_diffspec(self, plot=True):
        """
        Generate the difference spectrum.

        Returns
        -------
        pd.DataFrame
            The difference spectrum.
        """
        if len(self.columns) == 1:
            warnings.warn("The NEXAFS data does not contain multiple angles.")
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

        if "Diff" not in self.columns:
            warnings.warn("Normalizing only the isotropic data. NEXAFS is likely only one angle.")
            self[r"$\beta_{iso}$"] /= (lb + ub) / 2
        else:
            try:
                self[[r"$\beta_{iso}$", r"$\beta_{xx}$", r"$\beta_{zz}$"]] /= (lb + ub) / 2
            except:
                warnings.warn("Normalizing only the isotropic data. NEXAFS is likely only one angle.")
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
            delta_xx, beta_xx = kkcalc(self.index.values, self[r"$\beta_{xx}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)
            self.xx = OpticalConstant(delta_xx, beta_xx)
            self[r"$\delta_{xx}$"] = delta_xx(self.index.values)

        if r"$\beta_{zz}$" in self.columns:
            delta_zz, beta_zz = kkcalc(self.index.values, self[r"$\beta_{zz}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)
            self.zz = OpticalConstant(delta_zz, beta_zz)
            self[r"$\delta_{zz}$"] = delta_zz(self.index.values)

        if r"$\beta_{iso}$" in self.columns:
            delta_iso, beta_iso = kkcalc(self.index.values, self[r"$\beta_{iso}$"].to_numpy(), self.density, self.molecular_name, *args, **kwargs)
            self.iso = OpticalConstant(delta_iso, beta_iso)
            self[r"$\delta_{iso}$"] = delta_iso(self.index.values)


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
        
        with open(self.__db / "db.json", "r+") as f:
            data = json.load(f)

            if self.molecular_name in data["data"]["nexafs"]:
                data["data"]["nexafs"].remove(f"{self.molecular_name}")
                data["ocs"].remove(f"{self.molecular_name}")
            
            data["data"]["nexafs"].append(f"{self.molecular_name}")
            data["ocs"].append(f"{self.molecular_name}")

            f.seek(0)
            json.dump(data, f, indent=4)


        parquet = self.__db / ".data"/ "nexafs" / f"{self.molecular_name}.parquet"
        nexafs = self.__db / ".data"/ "nexafs" / f"{self.molecular_name}.nexafs"
        ocs = self.__db / ".ocs" / f"{self.molecular_name}.oc"
        
        
        self.to_parquet(parquet)
        self[self.angles.split(" ")].to_csv(nexafs)

        if self.angles == "55":
            optical_model = {
                "iso": self.iso,
            }
        else:
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
