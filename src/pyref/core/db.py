
import json
import pickle
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd
from scipy.interpolate import interp1d

from .frame import OpticalConstant


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
    

    
    