import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pyref import get_data_path

try:
    from pyref.pyref import py_get_image_for_row, py_read_fits_headers_only
except ImportError:
    py_get_image_for_row = None
    py_read_fits_headers_only = None

if py_get_image_for_row is None or py_read_fits_headers_only is None:
    raise NotImplementedError(
        "Image continuity check requires py_get_image_for_row (Phase 2). "
        "Run after Phase 2 is implemented."
    )

suffix = os.getenv("ANALYSIS_SUFFIX", "")
save_all = bool(suffix)
data_dir = get_data_path()
fits_files = list(data_dir.glob("*.fits"))

for i, fits_file in enumerate(fits_files):
    df = py_read_fits_headers_only(str(fits_file), ["Beamline Energy", "DATE", "EXPOSURE"])
    img, _ = py_get_image_for_row(df, 0)
    img = np.asarray(img)
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
