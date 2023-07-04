import os
from pathlib import Path
from shutil import copy2
from tqdm.auto import tqdm
from tkinter import *
from astropy.io import fits
from toolkit import *


def restartable(func):
    def wrapper(*args, **kwargs) -> None:
        answer = "y"
        while answer == "y":
            func(*args, **kwargs)
            while True:
                answer = input("Restart?  y/n:")
                if answer in ("y", "n"):
                    break
                else:
                    print("invalid answer")

    return wrapper


def check_parent(dir: Path) -> None:
    """
    Makes a new directory for the sorted data

    Parameters
    ----------
    dir : pathlib.Path
        Directory of the data that you want sorted
    """
    p_dir = dir.parent
    Directories = [x[0] for x in os.walk(p_dir)]
    sort_path = p_dir / "Sorted"
    if not sort_path.exists():
        sort_path.mkdir()
    else:
        print(
            "The sorted directory already exists - Checking for energy sub-directories"
        )
    return


# @restartable
def xrr_sorter(directory) -> None:
    """
    Collects the energies each fits was collected at and makes subfolder for each energy
    Generates a dictionary containing the current file location, and its destination.

    Parameters
    ----------
    dir : pathlib.Path
        Directory of the data that you want sorted
    """

    # check if the parent directory exist
    check_parent(directory)

    # Make a list of all fits files and their full path
    files = list(directory.glob("*fits"))
    sort_dir = directory.parent / "Sorted"

    for i, file in tqdm(enumerate(files)):
        # Opens the file nad reads the energy; round the energy to the nearest energy resolvable by the device
        with fits.open(file) as headers:
            new_en = round(headers[0].header[49], 1)  # type: ignore

        # determine the ending location
        dest = sort_dir / str(new_en)

        # makes a new directory for the new energy
        if not dest.exists():
            dest.mkdir()

        # copies the file to the new directory
        copy2(file, dest)
    return


if __name__ == "__main__":
    xrr_sorter(file_dialog())
