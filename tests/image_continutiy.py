import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pyref.pyref import py_read_fits

# grab all the fits files in the tests/data directory and read them
# it the images are not square, then generate a plot of them

suffix = os.getenv("ANALYSIS_SUFFIX", "")
save_all = bool(suffix)
data_dir = Path("/home/hduva/projects/pyref/tests/data")
fits_files = list(data_dir.glob("*.fits"))

for i, fits_file in enumerate(fits_files):
    df = py_read_fits(str(fits_file), ["Beamline Energy", "DATE", "EXPOSURE"])
    img = df["RAW"].to_numpy()[0]
    # compare to the astropy image
    with fits.open(fits_file) as hdul:
        astropy_img = hdul[2].data  # pyright: ignore[reportAttributeAccessIssue]
    print(img.shape, astropy_img.shape)
    if np.allclose(img, astropy_img) and not save_all:
        continue
    if save_all or i%20 == 0: # print every 20th image
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title("pyref image")
        axs[1].imshow(astropy_img)
        axs[1].set_title("astropy image")
        png_dir = data_dir / f"png{suffix}"
        png_dir.mkdir(exist_ok=True)
        plt.savefig(png_dir / (fits_file.stem + ".png"))
        plt.close()
        # save the images as numpy arrays allong with their differences
        # package them into a single numpy array to save as npz file
        npz_dir = data_dir / f"npz{suffix}"
        npz_dir.mkdir(exist_ok=True)
        np.savez(
            npz_dir / (fits_file.stem + ".npz"),
            pyref_img=img,
            astropy_img=astropy_img,
            diff=np.abs(img - astropy_img),
        )
