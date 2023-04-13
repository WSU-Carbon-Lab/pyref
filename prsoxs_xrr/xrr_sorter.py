import os, tkinter, glob, numpy as np
from pathlib import Path
from astropy.io import fits
from shutil import copy2
from tqdm import tqdm
from tkinter import *


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


def check_parent(dir) -> None:
    """
    Makes a new directory for the sorted data

    Parameters
    ----------
    dir : Windows path object
        Directory of the data that you want sorted
    """
    p_dir = Path(dir.parent)
    Directories = [x[0] for x in os.walk(p_dir)]
    sort_path = os.path.join(p_dir, "Sorted")
    if not os.path.exists(sort_path):
        os.makedirs(sort_path)
    else:
        print(
            "The sorted directory already exists - Checking for energy sub-directories"
        )
        pass
    return


@restartable
def xrr_sorter() -> None:
    """
    Collects the energies each fits was collected at and makes subfolder for each energy
    Generates a dictionary containing the current file location, and its destination.

    Parameters
    ----------
    dir : Windows path object
        Directory of the data that you want sorted
    """

    # get the data directory from user input using tkinter
    from tkinter import filedialog

    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    dir = Path(directory)

    # check if the parent directory exist
    check_parent(dir)

    # Make a list of all fits files and their full path
    files = list(dir.glob("*fits"))
    sort_dir = str(dir.parent) + "\Sorted"

    i = 0
    for file in tqdm(files):
        # Opens the file nad reads the energy; round the energy to the nearest energy resolvable by the device
        headers = fits.open(file)
        new_en = round(headers[0].header[49], 1)
        headers.close()

        # determine the ending location
        dest = os.path.join(str(sort_dir), str(new_en))

        # makes a new directory for the new energy
        if not os.path.exists(dest):
            os.makedirs(dest)
        else:
            pass

        # copies the file to the new directory
        copy2(file, dest)
        i += 1
    return


if __name__ == "__main__":
    xrr_sorter()
