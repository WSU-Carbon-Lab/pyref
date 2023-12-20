"""
ReflDataFrame
-------------
A subclass of pandas.DataFrame that contains 2d Data, metadata, and 
associated methods for working with reflectometry data.

@Author: Harlan Heilman
"""

import warnings
from calendar import c
from pathlib import Path
from typing import Literal, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kkcalc import data, kk
from matplotlib.pylab import beta
from periodictable.xsf import index_of_refraction

from .io import *

ArrayLike = Union[np.ndarray, list]
PathLike = Union[Path, str, NexafsIO]


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
        *args,
        **kwargs,
    ):
        io = NexafsIO(nexafs)
        df = io.get_nexafs(angles=angles)
        self.__dict__.update(df.__dict__)

        self.io = io
        self.angs = " ".join(angles)
        self.molecular_name = molecular_name
        self.density = density

    def __repr__(self):
        rep = super().__repr__()
        rep += "\n"
        rep += f"\nMolecular Name: {self.molecular_name}"
        rep += f"\nPath: {self.io.__repr__()}"
        rep += f"\nDensity: {self.density}"
        rep += f"\nAngles: {self.angs}"
        return rep

    def __str__(self):
        return self.__repr__()

    # ----------------------------------------------------------------
    # methods
    def plot_ds(self, *args, **kwargs):
        fig, ax = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0}
        )
        angs_ = self.angs.split(" ")
        angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]
        colors = sns.color_palette("coolwarm", len(angs))
        print(angs)
        self.plot(
            ax=ax[0],
            y=angs,
            color=colors,
            *args,
            **kwargs,
        )
        self.plot(ax=ax[1], y="Diff", legend=False, *args, **kwargs)

        ax[0].set(
            ylabel="Nexafs [a.u.]",
            title="ZnPc Optical Constants",
        )
        ax[0].legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
        )
        ax[1].set(ylabel="Diff", xlabel="Energy [eV]")

    def plot_ar(self, *args, **kwargs):
        angs = self.angs.split(" ")

        colors = sns.color_palette("coolwarm", len(angs))

        fig, ax = plt.subplots(
            nrows=1,
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        super().plot(
            y=angs, ax=ax, ylabel="Nexafs [a.u]", title="ZnPC Nexafs", color=colors
        )
        ax.legend(
            title="Angle [deg]",
            labels=[rf"${a}^\circ$" for a in angs],
            *args,
            **kwargs,
        )

    def plot_ooc(self, *args, **kwargs):
        angs_ = self.angs.split(" ")
        angs = [angs_[0], angs_[len(angs_) // 2], angs_[-1]]
        colors = sns.color_palette("coolwarm", len(angs) - 1)

        try:
            fig, ax = plt.subplots(
                nrows=3,
                sharex=True,
                gridspec_kw={"height_ratios": [2, 3, 1], "hspace": 0},
            )
            self.plot(ax=ax[0], y=angs, color=colors, legend=False)
            self.plot(ax=ax[2], y="Diff", legend=False)
            self.plot(
                ax=ax[1],
                y=[r"$\beta_{zz}$", r"$\beta_{iso}$", r"$\beta_{xx}$"],
                color=colors,
            )
            ax[0].set(ylabel="Nexafs [a.u.]", title="ZnPc Optical Constants")
            ax[0].legend(
                title="Angle [deg]",
                labels=[rf"${a}^\circ$" for a in angs],
                loc="upper right",
            )
            ax[1].set(ylabel="Optical Constants [a.u.]")
            ax[2].set(ylabel="Diff [a.u.]", xlabel="Energy [eV]")
        except:
            print("Generate the out of plane spectrum first using df.get_ooc()")

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
        xsf = (iso_intensity) / diff_intensity

        self[r"$\beta_{zz}$"] += zsf * self["Diff"]
        self[r"$\beta_{xx}$"] -= xsf * self["Diff"]
        self[r"$\beta_{iso}$"] = self["55"]

        if plot:
            self.plot_ooc()

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

    def run_kkcalc(self):
        """
        Run the KKCalc algorithm. The algorithm requires the absorbtion
        to be packed into a series of numpy arrays of collumns
        [Energy, Absobtion]. We apply this algorithm only to the
        amorphous, zz, and xx optical constants. The algorithm returns
        the atomic scattering factors that are then used to calculate
        the index of refraction. This algorithm also extends the energy
        range.

        Returns
        -------
        pd.DataFrame
            The dataframe with the calculated optical constants.
        """
        # get absorbtion arrays
        self.get_bare_atom()
        betas = [r"$\beta_{iso}'$", r"$\beta_{zz}'$", r"$\beta_{xx}'$"]
        deltas = [r"$\delta_{iso}'$", r"$\delta_{zz}'$", r"$\delta_{xx}'$"]

        nexafs = [
            np.column_stack((self.index.values, self["55"].to_numpy())),
            np.column_stack((self.index.values, self[r"$\beta_{zz}$"].to_numpy())),
            np.column_stack((self.index.values, self[r"$\beta_{xx}$"].to_numpy())),
        ]
        x_min = self.index.min()
        x_max = self.index.max()

        kdfs = []

        # run kkcalc
        for i, (delta, beta) in enumerate(zip(deltas, betas)):
            kdf = pd.DataFrame(columns=["Energy", delta, beta])
            out = kk.kk_calculate_real(
                nexafs[i],
                self.molecular_name,
                load_options=None,
                input_data_type="Beta",
                merge_points=[x_min, x_max],
                add_background=False,
                fix_distortions=False,
                curve_tolerance=0.05,
                curve_recursion=100,
            )
            kdf["Energy"] = out[:, 0]
            kdf.set_index("Energy", inplace=True)
            kdf[delta] = out[:, 1]
            kdf[beta] = out[:, 2]
            kdfs.append(kdf)
        return kdfs


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

    def __init__(self, raw_images: ReflIO = None, meta_data=None, *args, **kwargs):
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
