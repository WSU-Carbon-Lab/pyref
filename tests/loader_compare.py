from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import ImageHDU
from typing import cast

from pyref import get_data_path
from pyref.io import read_fits


@dataclass(frozen=True)
class PatchReport:
    """Patch-level comparison for a hotspot."""

    file: str
    hotspot_y: int
    hotspot_x: int
    pyref_value: float
    astropy_value: float
    diff_value: float
    patch_pyref: np.ndarray
    patch_astropy: np.ndarray
    patch_diff: np.ndarray


def load_pyref(path: Path) -> np.ndarray:
    """Load RAW via pyref."""
    df = read_fits(str(path), headers=[], engine="polars")
    return np.asarray(df["RAW"].to_numpy()[0])


def load_astropy(path: Path) -> np.ndarray:
    """Load RAW via astropy."""
    with fits.open(path) as hdul:
        image = cast(ImageHDU, hdul[2])
        return np.asarray(image.data)


def compare(path: Path) -> PatchReport:
    """Compare pyref and astropy outputs for one file."""
    pyref_img = load_pyref(path)
    astro_img = load_astropy(path)
    diff = pyref_img.astype(np.float64) - astro_img.astype(np.float64)
    flat_idx = int(np.argmax(np.abs(diff)))
    y, x = np.unravel_index(flat_idx, diff.shape)
    y = int(y)
    x = int(x)
    y0 = max(y - 2, 0)
    x0 = max(x - 2, 0)
    y1 = min(y + 3, diff.shape[0])
    x1 = min(x + 3, diff.shape[1])
    return PatchReport(
        file=path.name,
        hotspot_y=int(y),
        hotspot_x=int(x),
        pyref_value=float(pyref_img[y, x]),
        astropy_value=float(astro_img[y, x]),
        diff_value=float(diff[y, x]),
        patch_pyref=pyref_img[y0:y1, x0:x1],
        patch_astropy=astro_img[y0:y1, x0:x1],
        patch_diff=diff[y0:y1, x0:x1],
    )


def describe(path: Path) -> None:
    """Emit comparison details."""
    report = compare(path)
    print(report.file)
    print(f" hotspot (y={report.hotspot_y}, x={report.hotspot_x}) diff={report.diff_value:.0f}")
    print(f" pyref value {report.pyref_value:.0f} astropy value {report.astropy_value:.0f}")
    print(" patch pyref:")
    print(report.patch_pyref)
    print(" patch astropy:")
    print(report.patch_astropy)
    print(" patch diff:")
    print(report.patch_diff)
    print("-" * 40)


def main() -> None:
    """Run comparisons on selected FITS files."""
    base = get_data_path()
    targets = [
        base / "monlayerjune 81041-00001.fits",
        base / "monlayerjune 81041-00325.fits",
        base / "monlayerjune 81041-00225.fits",
    ]
    for path in targets:
        describe(path)


if __name__ == "__main__":
    main()
