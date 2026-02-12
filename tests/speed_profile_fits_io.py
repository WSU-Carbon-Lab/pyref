"""
Speed profile: header-only FITS scan.

Compares Rust (pyref) vs Astropy for header-only read (no pixel I/O).
Full read has been retired; use scan_experiment + get_image for on-demand images.
Searches for FITS in python/pyref/data then tests/fixtures.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

HEADER_KEYS = [
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
]

REPEAT = 5


def astropy_headers_only(path_strs: list[str], header_keys: list[str]) -> None:
    from astropy.io import fits

    for path in path_strs:
        with fits.open(path, lazy_load_hdus=True) as hdul:
            primary = hdul[0]
            for key in header_keys:
                primary.header.get(key)
            for hdu in hdul[1:]:
                if isinstance(hdu, fits.ImageHDU):
                    _ = hdu.header
                    break


def discover_fits(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".fits":
            out.append(root.resolve())
            continue
        for p in sorted(root.rglob("*.fits")):
            if p.is_file():
                out.append(p.resolve())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed profile: full vs header-only FITS read")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N FITS files (avoids schema mismatch for full read)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT,
        help=f"Repeat each mode N times (default {REPEAT})",
    )
    parser.add_argument(
        "--header-only",
        action="store_true",
        help="Time only header-only read (e.g. when full read fails on heterogeneous FITS)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    data_dir = repo / "python" / "pyref" / "data"
    fixtures_dir = repo / "tests" / "fixtures"

    paths = discover_fits(data_dir, fixtures_dir)
    if not paths:
        print(f"No FITS files found under {data_dir} or {fixtures_dir}")
        return

    if args.limit is not None:
        paths = paths[: args.limit]
        print(f"Limited to first {len(paths)} FITS files")
    path_strs = [str(p) for p in paths]
    print(f"FITS directory: {paths[0].parent}")
    print(f"FITS count: {len(paths)}")
    print(f"Repeat each mode {args.repeat} times\n")

    from pyref.pyref import py_read_multiple_fits_headers_only

    header_only_times: list[float] = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        try:
            py_read_multiple_fits_headers_only(path_strs, HEADER_KEYS)
        except Exception as e:
            print(f"Header-only read failed: {e}")
            header_only_times = []
            break
        header_only_times.append(time.perf_counter() - t0)
    header_only_mean = (
        sum(header_only_times) / len(header_only_times) if header_only_times else 0.0
    )

    astropy_headers_times: list[float] = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        try:
            astropy_headers_only(path_strs, HEADER_KEYS)
        except Exception as e:
            print(f"Astropy header-only failed: {e}")
            astropy_headers_times = []
            break
        astropy_headers_times.append(time.perf_counter() - t0)
    astropy_headers_mean = (
        sum(astropy_headers_times) / len(astropy_headers_times)
        if astropy_headers_times
        else 0.0
    )

    print("Rust (pyref) header-only:")
    if header_only_times:
        print(
            f"  mean = {header_only_mean*1000:.2f} ms  (min = {min(header_only_times)*1000:.2f} ms)"
        )
    print("Astropy (astropy.io.fits) header-only:")
    if astropy_headers_times:
        print(
            f"  mean = {astropy_headers_mean*1000:.2f} ms  (min = {min(astropy_headers_times)*1000:.2f} ms)"
        )
    if header_only_times and astropy_headers_times and astropy_headers_mean > 0:
        print(
            f"\nRust vs Astropy (header-only): Rust is {astropy_headers_mean/header_only_mean:.2f}x faster"
        )


if __name__ == "__main__":
    main()
