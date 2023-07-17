from typing import Final
import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk

HC: Final[int] = 12400


class FileDialog:
    @staticmethod
    def getDirectory() -> Path:
        root = Tk()
        root.withdraw()
        root.focus_force()
        directory = Path(filedialog.askdirectory())
        return directory

    @staticmethod
    def getFileName() -> Path:
        root = Tk()
        root.withdraw()
        root.focus_force()
        saveName = Path(filedialog.asksaveasfilename())
        return saveName

    @staticmethod
    def openFile() -> Path:
        root = Tk()
        root.withdraw()
        root.focus_force()
        openName = Path(filedialog.askopenfilename())
        return openName


class XrayDomainTransform:
    @staticmethod
    @np.vectorize
    def toLam(energy: float) -> float:
        global HC
        return HC / energy

    @staticmethod
    @np.vectorize
    def toK(energy: float) -> float:
        lam = XrayDomainTransform.toLam(energy)
        return 2 * np.pi / lam

    @staticmethod
    @np.vectorize
    def toQ(energy: float, twoTheta: float) -> float:
        lam = XrayDomainTransform.toLam(energy)
        return round(4 * np.pi * np.sin(np.radians(twoTheta)) / lam, 4)
