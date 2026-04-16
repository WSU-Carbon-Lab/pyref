r"""CI-friendly benchmark for ``pyref.catalog.ingest_beamtime``.

Generates ``--scans`` scans of ``--frames-per-scan`` FITS files at
``--width``x``--height`` pixels under a temporary beamtime root, ingests them
into an isolated catalog + cache, and prints the same markdown timing table
used by ``scripts/profile_beamtime_ingest.py``.

Because no real beamtime is required, this is safe for CI perf smoke tests and
local regression tracking. Typical usage from the repo root::

    uv run python scripts/bench_ingest.py --scans 10 --frames-per-scan 10 \
        --width 1024 --height 1024
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

warnings.filterwarnings("ignore", category=VerifyWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ingest_profile import (  # noqa: E402
    isolated_catalog_env,
    render_markdown_table,
    run_ingest_with_profile,
)

SYNTHETIC_SAMPLE_NAME = "synth"
BZERO_UNSIGNED_I16 = 32_768


def _build_primary_header(scan_idx: int, frame_idx: int) -> fits.Header:
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = 16
    hdr["NAXIS"] = 0
    hdr["DATE"] = "2024-02-02T00:00:00"
    hdr["Beamline Energy"] = 250.0 + float(scan_idx)
    hdr["Sample Theta"] = 1.0 + float(frame_idx) * 0.01
    hdr["CCD Theta"] = 2.0
    hdr["EPU Polarization"] = 1.0
    hdr["Higher Order Suppressor"] = 0.0
    return hdr


def _build_image_hdu(
    width: int,
    height: int,
    scan_idx: int,
    frame_idx: int,
) -> fits.ImageHDU:
    rng = np.random.default_rng(seed=scan_idx * 1_000_003 + frame_idx)
    raw = rng.integers(low=0, high=1024, size=(height, width), dtype=np.int32)
    data_i16 = (raw - BZERO_UNSIGNED_I16).astype(np.int16)
    hdu = fits.ImageHDU(data=data_i16)
    hdu.header["BZERO"] = BZERO_UNSIGNED_I16
    return hdu


def _write_synthetic_fits(
    path: Path,
    width: int,
    height: int,
    scan_idx: int,
    frame_idx: int,
) -> None:
    primary = fits.PrimaryHDU(header=_build_primary_header(scan_idx, frame_idx))
    image = _build_image_hdu(width, height, scan_idx, frame_idx)
    hdul = fits.HDUList([primary, image])
    hdul.writeto(path, overwrite=True)


def build_synthetic_beamtime(
    tmp_dir: Path,
    scans: int,
    frames_per_scan: int,
    width: int,
    height: int,
) -> Path:
    """Generate ``scans * frames_per_scan`` ingestible FITS files under ``tmp_dir``.

    Parameters
    ----------
    tmp_dir : pathlib.Path
        Parent directory for the synthetic beamtime layout.
    scans : int
        Number of synthetic scans to generate.
    frames_per_scan : int
        Frames per scan.
    width, height : int
        Pixel dimensions written to NAXIS1 and NAXIS2.

    Returns
    -------
    pathlib.Path
        Absolute path of the beamtime root (its ``CCD`` subdirectory holds frames).
    """
    beamtime = (tmp_dir / "beamtime").resolve()
    ccd_dir = beamtime / "CCD"
    ccd_dir.mkdir(parents=True, exist_ok=True)
    for scan_idx in range(scans):
        scan_number = scan_idx + 1
        for frame_idx in range(frames_per_scan):
            frame_number = frame_idx + 1
            stem = f"{SYNTHETIC_SAMPLE_NAME}-{scan_number:05d}-{frame_number:05d}"
            _write_synthetic_fits(
                ccd_dir / f"{stem}.fits",
                width,
                height,
                scan_idx,
                frame_idx,
            )
    return beamtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scans",
        type=int,
        default=10,
        help="Number of synthetic scans.",
    )
    parser.add_argument(
        "--frames-per-scan",
        type=int,
        default=10,
        help="Frames generated per synthetic scan.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="NAXIS1 for each synthetic frame.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="NAXIS2 for each synthetic frame.",
    )
    return parser.parse_args()


def main() -> None:
    """Build a synthetic beamtime, run ingest, and print a markdown timing report."""
    args = _parse_args()
    if args.scans <= 0 or args.frames_per_scan <= 0:
        msg = "--scans and --frames-per-scan must be positive integers"
        raise SystemExit(msg)
    if args.width <= 0 or args.height <= 0:
        msg = "--width and --height must be positive integers"
        raise SystemExit(msg)

    total_files = args.scans * args.frames_per_scan

    with tempfile.TemporaryDirectory(prefix="pyref-bench-fixture-") as fixture_dir:
        beamtime = build_synthetic_beamtime(
            Path(fixture_dir),
            args.scans,
            args.frames_per_scan,
            args.width,
            args.height,
        )
        with isolated_catalog_env():
            _catalog_path, profile = run_ingest_with_profile(beamtime)

    print(
        f"Synthetic beamtime: {args.scans} scans x {args.frames_per_scan} frames "
        f"({args.width}x{args.height} px) => {total_files} files"
    )
    print(f"FITS files (layout): {profile.layout_files}")
    cr = profile.counts["catalog_row"]
    fc = profile.counts["file_complete"]
    print(f"catalog_row events: {cr}  file_complete: {fc}")
    first_cr = profile.first_catalog_row_seconds
    first_fc = profile.first_file_complete_seconds
    if first_cr is not None:
        print(f"Time to first catalog_row: {first_cr:.3f} s")
    if first_fc is not None:
        print(f"Time to first file_complete: {first_fc:.3f} s")
    print()
    print(render_markdown_table(profile))
    if profile.wall_seconds > 0 and total_files > 0:
        fps = total_files / profile.wall_seconds
        wall_s = profile.wall_seconds
        print(
            f"files_per_second: {fps:.2f} "
            f"({total_files} files in {wall_s:.3f} s)"
        )


if __name__ == "__main__":
    main()
