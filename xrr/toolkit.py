from typing import Final
import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk

HC: Final[int] = 12400


class FileDialog:
    @staticmethod
    def getDirectory() -> Path | None:
        root = Tk()
        root.withdraw()
        root.focus_force()
        directory = Path(filedialog.askdirectory())
        return directory if directory else None

    @staticmethod
    def getFileName() -> Path | None:
        root = Tk()
        root.withdraw()
        root.focus_force()
        saveName = Path(filedialog.asksaveasfilename())
        return saveName if saveName else None

    @staticmethod
    def openFile() -> Path | None:
        root = Tk()
        root.withdraw()
        root.focus_force()
        openName = Path(filedialog.askopenfilename())
        return openName if openName else None


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
