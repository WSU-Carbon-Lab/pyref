from pathlib import Path
import pandas as pd
import numpy as np
from xrr._config import FILE_NAMES



class Reuse:
    @staticmethod
    def saveForReuse(obj):
        energy = obj.refl.Energy[0]
        pol = obj.refl.POL[0]
        saveDir = str(obj.path.parent) + f"{energy}_{pol}"

        obj.refl.to_csv(str(saveDir) + FILE_NAMES["df"], index = False)
        np.savez(str(saveDir) + FILE_NAMES["images"], obj.images)
        np.savez(str(saveDir) + FILE_NAMES["masked"], obj.masked)
        np.savez(str(saveDir) + FILE_NAMES["filtered"], obj.filtered)
        np.savez(str(saveDir) + FILE_NAMES["beamspot"], obj.beamspot)
        np.savez(str(saveDir) + FILE_NAMES["background"], obj.background)

    @staticmethod
    def openForReuse(obj):
        energy, pol = obj.path.parts[-2:]
        openDir = str(obj.path.parent) + f"{energy}_{pol}"

        obj.refl = pd.read_csv(openDir + FILE_NAMES["df"])
        imagedata = np.load(openDir + FILE_NAMES["df"])
        maskddata = np.load(openDir + FILE_NAMES["df"])
        filterdata = np.load(openDir + FILE_NAMES["df"])
        beamspotdata = np.load(openDir + FILE_NAMES["df"])
        backgrounddata = np.load(openDir + FILE_NAMES["df"])

        obj.images = [imagedata[key] for key in imagedata.files]
        obj.masked = [maskddata[key] for key in maskddata.files]
        obj.filtered = [filterdata[key] for key in filterdata.files]
        obj.beamspot = [beamspotdata[key] for key in beamspotdata.files]
        obj.background = [backgrounddata[key] for key in backgrounddata.files]
