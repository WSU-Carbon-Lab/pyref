from pathlib import Path
import pandas as pd
import numpy as np

FILE_NAMES = {
    "df": ".csv",
    "images": "_images.npz",
    "masked": "_masked.npz",
    "filtered": "_filtered.npz",
    "beamspot": "_beamspot.npz",
    "background": "_background.npz",
}


class Reuse:
    @staticmethod
    def saveForReuse(saveDir, df, images, masked, filtered, beam, background):
        df.to_csv(str(saveDir) + FILE_NAMES["df"])
        np.savez(str(saveDir) + FILE_NAMES["images"], images)
        np.savez(str(saveDir) + FILE_NAMES["masked"], masked)
        np.savez(str(saveDir) + FILE_NAMES["filtered"], filtered)
        np.savez(str(saveDir) + FILE_NAMES["beamspot"], beam)
        np.savez(str(saveDir) + FILE_NAMES["background"], background)

    @staticmethod
    def openForReuse(dataDir: Path):
        pol = dataDir.name
        en = dataDir.parent.name
        namePrefix = f'{en}_{pol}'
        openDir = dataDir.parent.parent

        df = pd.read_csv(str(openDir) + namePrefix + FILE_NAMES["df"], index_col=0)
        imagedata = np.load(str(openDir) + namePrefix + FILE_NAMES["images"])
        maskddata = np.load(str(openDir) + namePrefix + FILE_NAMES["masked"])
        filterdata = np.load(str(openDir) + namePrefix + FILE_NAMES["filtered"])
        beamspotdata = np.load(str(openDir) + namePrefix + FILE_NAMES["beamspot"])
        backgrounddata = np.load(str(openDir) + namePrefix + FILE_NAMES["background"])

        images = [imagedata[key] for key in imagedata.files]
        masks = [maskddata[key] for key in maskddata.files]
        filtered = [filterdata[key] for key in filterdata.files]
        beam = [beamspotdata[key] for key in beamspotdata.files]
        background = [backgrounddata[key] for key in backgrounddata.files]
        return df, images, masks, filtered, beam, background
