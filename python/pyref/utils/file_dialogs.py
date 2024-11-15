"""File Dialogs."""

from pathlib import Path
from tkinter import Tk, filedialog
from typing import Final

HC: Final[int] = 12400


class FileDialog:
    @staticmethod
    def _createDialog() -> Tk:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # Places the dialog at the top
        return root

    @staticmethod
    def getDirectory(title: str | None = None, *args, **kwargs) -> Path:
        root = FileDialog._createDialog()
        directory = Path(
            filedialog.askdirectory(title=title, parent=root, *args, **kwargs)
        )
        root.destroy()
        return directory

    @staticmethod
    def getFileName(title: str | None = None, *args, **kwargs) -> Path:
        root = FileDialog._createDialog()
        saveName = Path(
            filedialog.asksaveasfilename(title=title, parent=root, *args, **kwargs)
        )
        root.destroy()
        return saveName

    @staticmethod
    def openFile(title: str | None = None, *args, **kwargs) -> Path:
        root = FileDialog._createDialog()
        openName = Path(
            filedialog.askopenfilename(title=title, parent=root, *args, **kwargs)
        )
        root.destroy()
        return openName


# class XrayDomainTransform:
#     @staticmethod
#     @np.vectorize
#     def toLam(energy: float) -> float:
#         global HC
#         return HC / energy

#     @staticmethod
#     @np.vectorize
#     def toK(energy: float) -> float:
#         lam = XrayDomainTransform.toLam(energy)
#         return 2 * np.pi / lam

#     @staticmethod
#     @np.vectorize
#     def toQ(energy: float, twoTheta: float) -> float:
#         lam = XrayDomainTransform.toLam(energy)
#         return round(4 * np.pi * np.sin(np.radians(twoTheta)) / lam, 4)


if __name__ == "__main__":
    FileDialog.getDirectory(title="Test")
