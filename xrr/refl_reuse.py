from pathlib import Path
import pandas as pd
import numpy as np
import fastparquet
try:
    from xrr._config import FILE_NAMES
except:
    from _config import FILE_NAMES


class Reuse:
    @staticmethod
    def saveForReuse(obj):
        energy = round(obj.refl.Energy[0],1)
        pol = obj.refl.POL[0]
        saveDir = str(obj.path.parent.parent / f"{energy}_{pol}")

        images = obj.refl.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        images.to_parquet(str(saveDir) + FILE_NAMES["image.parquet"], index = False, compression = 'gzip')
        obj.refl.to_parquet(str(saveDir) + FILE_NAMES["meta.parquet"], index = False, compression = 'gzip')
        obj.refl.to_json(str(saveDir) + FILE_NAMES[".json"], orient = "columns", compression = 'gzip')

    @staticmethod
    def openForReuse(obj):
        energy, pol = obj.path.parts[-2:]
        openDir = str(obj.path.parent.parent / f"{energy}_{pol}")

        images = fastparquet.ParquetFile(openDir + FILE_NAMES["image.parquet"])
        refl = fastparquet.ParquetFile(openDir + FILE_NAMES["meta.parquet"])
        obj.refl = refl.to_pandas()
        obj.images = images.to_pandas()
    
    @staticmethod
    def multiOpen(multiDir: Path, energy: str, pol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        openDir = str(multiDir / f'{energy}_{pol}')
        images = fastparquet.ParquetFile(openDir + FILE_NAMES["image.parquet"])
        refl = fastparquet.ParquetFile(openDir + FILE_NAMES["meta.parquet"])
        return refl.to_pandas(), images.to_pandas()