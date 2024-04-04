import datetime
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any


def get_xrr_directory(bt=None) -> Path:
    """
    Returns the current beamtime directory.

    Returns
    -------
    Path
        The current beamtime directory.
    """
    data_path = (
        Path("Washington State University (email.wsu.edu)")
        / "Carbon Lab Research Group - Documents"
        / "Synchrotron Logistics and Data"
        / "ALS - Berkeley"
        / "Data"
        / "BL1101"
    )
    date = datetime.datetime.now()
    if bt is None:
        bt = date.strftime("%Y%b")

    return Path.home() / data_path / bt / "XRR"


class FileDialog:
    """Class for creating a file dialog window."""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call method for the FileDialog class."""
        return self.getFileName(*args, **kwds)

    @staticmethod
    def _createDialog() -> Tk:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # Places the dialog at the top
        return root

    @staticmethod
    def getDirectory(title: str | None = None, *args, **kwargs) -> Path:
        """Get the selected directory from the file dialog."""
        root = FileDialog._createDialog()
        directory = Path(filedialog.askdirectory(title=title, parent=root, **kwargs))
        root.destroy()
        return directory

    @staticmethod
    def getFileName(title: str | None = None, *args, **kwargs) -> Path:
        """Get the selected file name from the file dialog."""
        root = FileDialog._createDialog()
        saveName = Path(
            filedialog.asksaveasfilename(title=title, parent=root, **kwargs)
        )
        root.destroy()
        return saveName

    @staticmethod
    def openFile(title: str | None = None, *args, **kwargs) -> Path:
        """Open the selected file from the file dialog."""
        root = FileDialog._createDialog()
        openName = Path(filedialog.askopenfilename(title=title, parent=root, **kwargs))
        root.destroy()
        return openName
