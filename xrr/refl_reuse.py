import pandas as pd
import numpy as np

FILE_NAMES = {
    "df": ".csv",
    "images": "images.npz",
    "masked": "masked.npz",
    "filtered": "filtered.npz",
    "beamspot": "beamspot.npz",
    "background": "background.npz",
}


class Reuse:
    @staticmethod
    def saveForReuse(saveDir, df, images, masked, filtered, beam, background):
        df.refl.to_csv(FILE_NAMES["df"])
        np.savez(str(saveDir) + FILE_NAMES["images"], images)
        np.savez(str(saveDir) + FILE_NAMES["masked"], masked)
        np.savez(str(saveDir) + FILE_NAMES["filtered"], filtered)
        np.savez(str(saveDir) + FILE_NAMES["beamspot"], beam)
        np.savez(str(saveDir) + FILE_NAMES["background"], background)

    @staticmethod
    def openForReuse(openDir):
        df = pd.read_csv(str(openDir) + "csv", index_col=0)
        imagedata = np.load(str(openDir) + FILE_NAMES["images"])
        maskddata = np.load(str(openDir) + FILE_NAMES["masked"])
        filterdata = np.load(str(openDir) + FILE_NAMES["filtered"])
        beamspotdata = np.load(str(openDir) + FILE_NAMES["beamspot"])
        backgrounddata = np.load(str(openDir) + FILE_NAMES["background"])

        images = [imagedata[key] for key in imagedata.files]
        masks = [maskddata[key] for key in maskddata.files]
        filtered = [filterdata[key] for key in filterdata.files]
        beam = [beamspotdata[key] for key in beamspotdata.files]
        background = [backgrounddata[key] for key in backgrounddata.files]
        return df, images, masks, filtered, beam, background


