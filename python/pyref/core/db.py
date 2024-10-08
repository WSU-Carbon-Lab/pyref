import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from refnx.dataset import ReflectDataset
from rich import print
from scipy.interpolate import interp1d

from pyref.core.config import AppConfig as config

# from pyref.core.frame import OpticalConstant
from pyref.core.paths import FileDialog


class db:
    """A class representing a database for storing data."""

    def __init__(self):
        self.db = config.DB
        self.data = self.db / ".data"
        self.nexafs = self.data / "nexafs"
        self.refl = self.data / "refl"
        self.res = self.data / "res"
        self.ocs = self.db / ".ocs"
        self.struct = self.db / ".struct"

    def __repr__(self) -> str:
        return f"pyref.db({self.db})"

    def init(self, path: str | Path) -> None:
        """
        Initialize a database for storing data.

        Parameters
        ----------
        path : str | Path
            The path to the database.
        """
        init_db(path)

    def add_df(
        self,
        df: pd.DataFrame,
        molecular_name: str,
        orientation: Literal["iso", "xx", "zz"],
    ) -> None:
        """
        Save a dataframe to the database.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        molecular_name : str
            The name of the molecule.
        orientation : Literal["iso", "xx", "zz"]
            The orientation of the molecule.
        """
        df_to_db(df, molecular_name, orientation)

    def query(self, molecular_name):
        """
        Query the database for a molecular name.

        Parameters
        ----------
        molecular_name : str
            The name of the molecule to query.

        Returns
        -------
        df : pd.DataFrame
            The dataframe of the molecule.
        """
        with (self.db / "db.json").open() as f:
            data = json.load(f)
        if molecular_name in data["data"]["nexafs"]:
            print(f"NEXAFS data located for {molecular_name}. :sparkles:")
        else:
            print(
                f"[bold red]NEXAFS data not located for {molecular_name}. [/bold red] :warning:"
            )

        if molecular_name in data["ocs"]:
            print(f"Optical constant model located for {molecular_name}. :sparkles:")
        else:
            print(
                f"[bold red]Optical constant model not located for {molecular_name}. [/bold red] :warning:"
            )

        structs = {}
        for struct in data["struct"]:
            if molecular_name in data["struct"][struct]:
                with (self.struct / f"{struct}.json").open() as f:
                    structs[struct] = json.load(f)
        if len(structs) > 0:
            print(f"Structures located for {molecular_name}. :sparkles:")
            print(structs)
        else:
            print(
                f"[bold red]Structures not located for {molecular_name}. [/bold red] :warning:"
            )

    def get_oc(self, molecular_name):
        """
        Get the optical constant model for a molecule.

        Parameters
        ----------
        molecular_name : str
            The name of the molecule to query.
        orientation : Literal["iso", "xx", "zz"]
            The orientation of the molecule.

        Returns
        -------
        oc : OpticalConstant
            The optical constant model.
        """
        with (self.ocs / f"{molecular_name}.oc").open("rb") as file:
            oc = pickle.load(file)
        return oc

    def get_nexafs(self, molecular_name):
        """
        Get the NEXAFS data for a molecule.

        Parameters
        ----------
        molecular_name : str
            The name of the molecule to query.

        Returns
        -------
        df : pd.DataFrame
            The dataframe of the molecule.
        """
        df = pd.read_parquet(self.nexafs / f"{molecular_name}.parquet")
        return df

    def get_struct(self, struct_name):
        """
        Get the structure for a molecule.

        Parameters
        ----------
        struct_name : str
            The name of the structure to query.

        Returns
        -------
        struct : dict
            The structure of the molecule.
        """
        with (self.struct / f"{struct_name}.json").open() as file:
            struct = json.load(file)
        return struct

    def set_struct(self, struct_name: str, struct: dict):
        """
        Set the structure for a molecule.

        Parameters
        ----------
        struct_name : str
            The name of the structure to query.
        struct : dict
            The structure of the molecule.
        """
        assert "layers" in struct, "Structure must have a 'layers' key."

        for layer in struct["layers"]:
            assert (
                "thickness" in struct[layer]
            ), f"Layer {layer} must have a 'thickness' key."
            assert (
                "roughness" in struct[layer]
            ), f"Layer {layer} must have a 'roughness' key."
            assert (
                "density" in struct[layer]
            ), f"Layer {layer} must have a 'density' key."

        struct_path = self.struct / f"{struct_name}.json"
        if struct_path.exists():
            w = f"Structure {struct_name} already exists. Overwriting."
            warnings.warn(w, stacklevel=2)
        else:
            struct_path.touch()

        with (self.db / "db.json").open("r+") as f:
            data = json.load(f)
            data["struct"][struct_name] = struct
            f.seek(0)
            json.dump(data, f, indent=4)

        with struct_path.open("r+") as f:
            json.dump(struct, f, indent=4)

    def get_refl(self, file_name: str | Path | None, sample: str, restat=True):
        """
        Get the reflectivity data for a molecule.

        Parameters
        ----------
        file_name : str | Path | None
            The name of the reflectivity file to query.
        sample : str
            The name of the sample.
        restat : bool, optional
            Whether to restat the reflectivity data, by default True.

        Returns
        -------
        df : pd.DataFrame
            The dataframe of the molecule.
        """
        if file_name is None:
            file_name = FileDialog().getFileName(
                title="Select reflectivity file", initialdir=self.refl
            )
        file_name = Path(file_name)
        if file_name.suffix == ".csv":
            refl = pd.read_csv(self.refl / f"{sample}" / f"{file_name}")
        elif file_name.suffix == ".parquet" or file_name.suffix == ".gzip":
            refl = pd.read_parquet(self.refl / f"{sample}" / f"{file_name}")
        else:
            e = f"File must be a .csv or .parquet file. received {file_name.suffix}"
            raise ValueError(e)

        if restat:
            refl["Err"] = 0.1 * refl["Refl"]

        return refl


def ensure_normalized(refl):
    """
    Ensure that the reflectivity data is normalized.

    Parameters
    ----------
    refl : pd.DataFrame
        The reflectivity data.

    Returns
    -------
    pd.DataFrame
        The normalized reflectivity data.
    """
    refl.Refl /= refl.Refl.iloc[0]
    return refl


def smart_mask(refl, pol: Literal["s", "p"] = "s"):
    """
    Apply a smart mask to the reflectivity data based on the polarization.

    Parameters
    ----------
    refl : pd.DataFrame
        The reflectivity data.
    pol : {"s", "p"}, optional
        The polarization type, by default "s".

    Returns
    -------
    pd.DataFrame
        The masked reflectivity data.
    """
    # refl = ensure_nromalized(refl)
    if pol == "p":
        # locate the brewster angle ~45 deg cutoff is chosen based on the
        # reflectivity at the brewster angle
        brewster_angles = (refl.Theta > 44) & (refl.Theta < 46)
        ba_int = refl[brewster_angles].Refl.mean()
        ba_cutoff = ba_int + 3 * refl[brewster_angles].Refl.std()
        refl = refl[refl.Refl > ba_cutoff]

    # drop points q<0.02
    refl = refl[refl.Q > 0.02]
    return refl


type percent = float


def to_refnx_dataset(
    refl,
    pol: Literal["s", "p", "sp"] = "s",
    second_pol: pd.DataFrame | None = None,
    error: percent | None = None,
):
    """
    Convert reflectivity data to a refnx ReflectDataset.

    Parameters
    ----------
    refl : pd.DataFrame
        The reflectivity data.
    pol : {"s", "p", "sp"}, optional
        The polarization type, by default "s".
    second_pol : pd.DataFrame or None, optional
        The second polarization data, by default None.
    error : float or None, optional
        The error percentage, by default None.

    Returns
    -------
    refnx.reflect.ReflectDataset
        The refnx ReflectDataset.
    """
    q = refl.Q.to_numpy()
    r = refl.Refl.to_numpy()
    _dr = refl.Err.to_numpy()
    if pol == "sp" and isinstance(second_pol, pd.DataFrame):
        # append the second polarization
        import numpy as np

        q = np.append(q, second_pol.Q.to_numpy())
        r = np.append(r, second_pol.Refl.to_numpy())
        _dr = np.append(_dr, second_pol.Err.to_numpy())
    if error is None:
        dr = _dr
    else:
        del _dr
        dr = error * r
    return ReflectDataset(data=(q, r, dr))


def init_db(path: str | Path) -> None:
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
    with config.open("w") as f:
        json.dump(config_json, f, indent=4)

    path.mkdir(parents=True, exist_ok=True)
    dbjson = {
        ".data": {
            "nexafs": [],
            "xrr": [],
        },
        ".ocs": [],
        ".struct": [],
    }
    with (path / "db.json").open("w") as f:
        json.dump(dbjson, f, indent=4)


def df_to_db(
    df: pd.DataFrame, molecular_name: str, orientation: Literal["iso", "xx", "zz"]
) -> None:
    """
    Save a dataframe to the database. The first col should be.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    molecular_name : str
        The name of the molecular data.
    orientation : {"iso", "xx", "zz"}
        The orientation of the data.

    Returns
    -------
    None
    """
    # construct the optical constant function
    cols = df.columns
    energy = df.index.to_numpy()
    delta = df[[col for col in cols if "delta" in col]].to_numpy()
    beta = df[[col for col in cols if "beta" in col]].to_numpy()

    delta_interp = interp1d(energy, delta, axis=0)
    beta_interp = interp1d(energy, beta, axis=0)
    oc = OpticalConstant(delta_interp, beta_interp)
    optical_model = {
        f"{orientation}": oc,
    }

    # update the database
    config = Path(__file__).parent / "config.json"
    with config.open() as f:
        __db = Path(json.load(f)["db"])
    with (__db / "db.json").open("r+") as f:
        data = json.load(f)

        if molecular_name in data["data"]["nexafs"]:
            data["data"]["nexafs"].remove(f"{molecular_name}")
            data["ocs"].remove(f"{molecular_name}")

        data["data"]["nexafs"].append(f"{molecular_name}")
        data["ocs"].append(f"{molecular_name}")

        f.seek(0)
        json.dump(data, f, indent=4)

    # save the data
    parquet = __db / ".data" / "nexafs" / f"{molecular_name}.parquet"
    nexafs = __db / ".data" / "nexafs" / f"{molecular_name}.nexafs"
    ocs = __db / ".ocs" / f"{molecular_name}.oc"

    with parquet.open("wb") as f:
        df.to_parquet(f)
    with nexafs.open("w") as f:
        df.to_csv(f)
    with ocs.open("wb") as f:
        pickle.dump(optical_model, f)
