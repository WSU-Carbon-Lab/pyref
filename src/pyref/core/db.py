
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from refnx.dataset import ReflectDataset
from rich import print
from scipy.interpolate import interp1d

from .frame import OpticalConstant
from .paths import FileDialog


class db:

    def __init__(self):
        with open(Path(__file__).parent / "config.json", "r") as f:
            path = json.load(f)["db"]
        self.db = Path(path)
        self.nexafs = self.db / ".data" / "nexafs"
        self.refl = self.db / ".data" / "refl"
        self.ocs = self.db / ".ocs"
        self.struct = self.db / ".struct"
    
    def __repr__(self) -> str:
        return f"pyref.db({self.db})"


    def init(self, path:str | Path) -> None:
        """
        Initialize a database for storing data.

        Parameters
        ----------
        path : str | Path
            The path to the database.
        """
        init_db(path)
    
    def add_df(self, df: pd.DataFrame, molecular_name, orientation : Literal["iso", "xx", "zz"]) -> None:
        """
        Save a dataframe to the database. The first col should be 

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
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
        with open(self.db / "db.json", "r") as f:
            data = json.load(f)
        if molecular_name in data["data"]["nexafs"]:
            print(f"NEXAFS data located for {molecular_name}. :sparkles:")
        else:
            print(f"[bold red]NEXAFS data not located for {molecular_name}. [/bold red] :warning:")

        if molecular_name in data["ocs"]:
            print(f"Optical constant model located for {molecular_name}. :sparkles:")
        else:
            print(f"[bold red]Optical constant model not located for {molecular_name}. [/bold red] :warning:")
        
        structs = {}
        for struct in data["struct"]:
            if molecular_name in data["struct"][struct]:
                with open(self.struct / f"{struct}.json", "r") as f:
                    structs[struct] = json.load(f)
        if len(structs) > 0:
            print(f"Structures located for {molecular_name}. :sparkles:")
            print(structs)
        else:
            print(f"[bold red]Structures not located for {molecular_name}. [/bold red] :warning:")

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
        oc = pickle.load(open(self.ocs / f"{molecular_name}.oc", "rb"))
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
        struct = json.load(open(self.struct / f"{struct_name}.json", "r"))
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
            assert "thickness" in struct[layer], f"Layer {layer} must have a 'thickness' key."
            assert "roughness" in struct[layer], f"Layer {layer} must have a 'roughness' key."
            assert "density" in struct[layer], f"Layer {layer} must have a 'density' key."
        
        struct_path = self.struct / f"{struct_name}.json"
        if struct_path.exists():
            warnings.warn(f"Structure {struct_name} already exists. Overwriting.")
        else:
            struct_path.touch()

        with open(self.db / "db.json", "r+") as f:
            data = json.load(f)
            data["struct"][struct_name] = struct
            f.seek(0)
            json.dump(data, f, indent=4)
        
        with open(self.struct / f"{struct_name}.json", "r+") as f:
            json.dump(struct, f, indent=4)
    
    def get_refl(self, file_name: str | Path | None, sample:str,restat = True):
        """
        Get the reflectivity data for a molecule.

        Parameters
        ----------
        file_name : str
            The name of the reflectivity file to query.

        Returns
        -------
        df : pd.DataFrame
            The dataframe of the molecule.
        """

        if file_name is None:
            file_name = FileDialog().getFileName(title="Select reflectivity file", initialdir=self.refl)

        file_name = Path(file_name)
        if file_name.suffix == ".csv":
            refl = pd.read_csv(self.refl / f"{sample}" /f"{file_name}")
        elif file_name.suffix == ".parquet" or file_name.suffix == ".gzip":
            refl = pd.read_parquet(self.refl / f"{sample}" /f"{file_name}")
        else:
            raise ValueError(f"File must be a .csv or .parquet file. recieved {file_name.suffix}")
        

        if restat:
            refl["Err"] = .1 * refl["Refl"]
        
        return refl

def ensure_nromalized(refl):
    refl.Refl /= refl.Refl.iloc[0]
    return refl

def smart_mask(refl, pol: Literal["s", "p"] = "s"):
    # refl = ensure_nromalized(refl)
    if pol == "p":
        # locate the brewster angle ~45 deg cutoff is chosen based on the 
        # reflectivity at the brewster angle
        brewster_angles = (44 < refl.Theta) & (refl.Theta < 46)
        ba_int = refl[brewster_angles].Refl.mean()
        ba_cutoff = ba_int + 3 * refl[brewster_angles].Refl.std()
        refl = refl[refl.Refl > ba_cutoff]
    
    # drop points q<0.02
    refl = refl[refl.Q > 0.02]
    return refl

def to_refnx_dataset(refl, pol: Literal["s", "p", "sp"] = "s", second_pol: pd.DataFrame | None = None):

    q = refl.Q.to_numpy()
    r = refl.Refl.to_numpy()
    if pol == "sp" and isinstance(second_pol, pd.DataFrame):
        # append the second polarization
        import numpy as np
        q = np.append(q, second_pol.Q.to_numpy())
        r = np.append(r, second_pol.Refl.to_numpy())
        
    dr = .1*r
    return ReflectDataset(data = (q, r, dr))

        


    


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


def df_to_db(df: pd.DataFrame, molecular_name, orientation: Literal["iso", "xx", "zz"]) -> None:
    """
    Save a dataframe to the database. The first col should be 

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
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
    with open(config, "r") as f:
        __db = Path(json.load(f)["db"])
    with open(__db / "db.json", "r+") as f:
        data = json.load(f)

        if molecular_name in data["data"]["nexafs"]:
            data["data"]["nexafs"].remove(f"{molecular_name}")
            data["ocs"].remove(f"{molecular_name}")
        
        data["data"]["nexafs"].append(f"{molecular_name}")
        data["ocs"].append(f"{molecular_name}")

        f.seek(0)
        json.dump(data, f, indent=4)

    
    # save the data
    parquet = __db / ".data"/ "nexafs" / f"{molecular_name}.parquet"
    nexafs = __db / ".data"/ "nexafs" / f"{molecular_name}.nexafs"
    ocs = __db / ".ocs" / f"{molecular_name}.oc"

    df.to_parquet(parquet)
    df.to_csv(nexafs)

    pickle.dump(optical_model, open(ocs, "wb"))
    

    
    