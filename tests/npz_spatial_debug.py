from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class Hotspot:
    """Hotspot location and value."""

    y: int
    x: int
    value: float


def find_hotspot(diff: np.ndarray) -> Hotspot:
    """Return hotspot coordinates and value."""
    flat_idx = int(np.argmax(diff))
    y, x = np.unravel_index(flat_idx, diff.shape)
    return Hotspot(y=int(y), x=int(x), value=float(diff[y, x]))


def render_debug(pyref_img: np.ndarray, astropy_img: np.ndarray, diff: np.ndarray, output: Path) -> None:
    """Render spatial diagnostics for a pair of images."""
    hotspot = find_hotspot(diff)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    im0 = axes[0, 0].imshow(pyref_img, cmap="magma", origin="lower")
    axes[0, 0].set_title("pyref RAW")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im1 = axes[0, 1].imshow(astropy_img, cmap="magma", origin="lower")
    axes[0, 1].set_title("astropy RAW")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im2 = axes[1, 0].imshow(diff, cmap="viridis", origin="lower")
    axes[1, 0].set_title(f"abs diff (hotspot {hotspot.value:.0f} at y={hotspot.y}, x={hotspot.x})")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    row = hotspot.y
    col = hotspot.x
    axes[1, 1].plot(pyref_img[row, :], label="pyref row")
    axes[1, 1].plot(astropy_img[row, :], label="astropy row", linestyle="--")
    axes[1, 1].plot(pyref_img[:, col], label="pyref col")
    axes[1, 1].plot(astropy_img[:, col], label="astropy col", linestyle="--")
    axes[1, 1].set_title("line profiles through hotspot")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("pixel index")
    axes[1, 1].set_ylabel("counts")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate spatial debug plots for all NPZ bundles."""
    from pyref import get_data_path

    suffix = os.getenv("ANALYSIS_SUFFIX", "")
    data_dir = get_data_path()
    npz_dir = data_dir / f"npz{suffix}"
    debug_dir = data_dir / f"debug{suffix}"
    for path in sorted(npz_dir.glob("*.npz")):
        data = np.load(path)
        pyref_img = np.asarray(data["pyref_img"])
        astropy_img = np.asarray(data["astropy_img"])
        diff = np.asarray(data["diff"])
        output = debug_dir / f"{path.stem}_debug.png"
        render_debug(pyref_img, astropy_img, diff, output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
