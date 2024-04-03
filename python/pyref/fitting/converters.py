import pandas as pd
from refnx.dataset import ReflectDataset


def csv_to_refl(csv_path):
    """
    Convert a csv file to a refl file.
    """
    df = pd.read_csv(csv_path)
    refl = ReflectDataset(df.to_numpy().T)
    return refl

def parquet_to_refl(parquet_path):
    """
    Convert a parquet file to a refl file.
    """
    df = pd.read_parquet(parquet_path)
    refl = ReflectDataset(df.to_numpy().T)
    return refl

