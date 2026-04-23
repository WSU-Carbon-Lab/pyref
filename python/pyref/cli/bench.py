"""
``pyref bench``: synthetic ingest harness and real-beamtime wall-clock profiling.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import typer
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from pyref.ingest_profile import (
    format_seconds,
    isolated_catalog_env,
    render_markdown_table,
    run_ingest_with_profile,
)

warnings.filterwarnings("ignore", category=VerifyWarning)

app = typer.Typer(help="Ingest benchmarking and wall-clock profiling")

_SYNTHETIC_SAMPLE_NAME = "synth"
_BZERO_UNSIGNED_I16 = 32_768


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
    data_i16 = (raw - _BZERO_UNSIGNED_I16).astype(np.int16)
    hdu = fits.ImageHDU(data=data_i16)
    hdu.header["BZERO"] = _BZERO_UNSIGNED_I16
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


def _build_synthetic_beamtime(
    tmp_dir: Path,
    scans: int,
    frames_per_scan: int,
    width: int,
    height: int,
) -> Path:
    beamtime = (tmp_dir / "beamtime").resolve()
    ccd_dir = beamtime / "CCD"
    ccd_dir.mkdir(parents=True, exist_ok=True)
    for scan_idx in range(scans):
        scan_number = scan_idx + 1
        for frame_idx in range(frames_per_scan):
            frame_number = frame_idx + 1
            stem = (
                f"{_SYNTHETIC_SAMPLE_NAME}-{scan_number:05d}-{frame_number:05d}"
            )
            _write_synthetic_fits(
                ccd_dir / f"{stem}.fits",
                width,
                height,
                scan_idx,
                frame_idx,
            )
    return beamtime


@app.command("synthetic")
def bench_synthetic(
    scans: int = typer.Option(10, "--scans", help="Number of synthetic scans."),
    frames_per_scan: int = typer.Option(
        10,
        "--frames-per-scan",
        help="Frames generated per synthetic scan.",
    ),
    width: int = typer.Option(1024, "--width", help="NAXIS1 for each synthetic frame."),
    height: int = typer.Option(
        1024,
        "--height",
        help="NAXIS2 for each synthetic frame.",
    ),
) -> None:
    """Generate a temporary beamtime, ingest into an isolated catalog, print timings."""
    if scans <= 0 or frames_per_scan <= 0:
        raise typer.BadParameter("--scans and --frames-per-scan must be positive")
    if width <= 0 or height <= 0:
        raise typer.BadParameter("--width and --height must be positive")

    total_files = scans * frames_per_scan

    with tempfile.TemporaryDirectory(prefix="pyref-bench-fixture-") as fixture_dir:
        beamtime = _build_synthetic_beamtime(
            Path(fixture_dir),
            scans,
            frames_per_scan,
            width,
            height,
        )
        with isolated_catalog_env():
            _catalog_path, profile = run_ingest_with_profile(beamtime)

    typer.echo(
        f"Synthetic beamtime: {scans} scans x {frames_per_scan} frames "
        f"({width}x{height} px) => {total_files} files"
    )
    typer.echo(f"FITS files (layout): {profile.layout_files}")
    cr = profile.counts["catalog_row"]
    fc = profile.counts["file_complete"]
    typer.echo(f"catalog_row events: {cr}  file_complete: {fc}")
    first_cr = profile.first_catalog_row_seconds
    first_fc = profile.first_file_complete_seconds
    if first_cr is not None:
        typer.echo(f"Time to first catalog_row: {first_cr:.3f} s")
    if first_fc is not None:
        typer.echo(f"Time to first file_complete: {first_fc:.3f} s")
    typer.echo()
    typer.echo(render_markdown_table(profile))
    if profile.wall_seconds > 0 and total_files > 0:
        fps = total_files / profile.wall_seconds
        wall_s = profile.wall_seconds
        typer.echo(
            f"files_per_second: {fps:.2f} "
            f"({total_files} files in {wall_s:.3f} s)"
        )


@app.command("profile")
def bench_profile(
    beamtime: Path = typer.Option(
        ...,
        "--beamtime",
        help="Beamtime root (e.g. ALS date folder containing CCD data).",
    ),
    use_default_paths: bool = typer.Option(
        False,
        "--use-default-paths",
        help="Do not override PYREF_CATALOG_DB / PYREF_CACHE_ROOT.",
    ),
) -> None:
    """Print ingest phase wall times for a real beamtime (Rust progress events)."""
    beam = beamtime.resolve()

    with isolated_catalog_env(
        enabled=not use_default_paths,
        prefix="pyref-ingest-profile-",
    ):
        _catalog_path, profile = run_ingest_with_profile(beam)

    typer.echo(f"Beamtime: {beam}")
    typer.echo(f"FITS files (layout): {profile.layout_files}")
    if profile.layout_files == 0:
        typer.echo(
            "Note: 0 files often means no ingestible `.fits` "
            "(stems starting with `_` are skipped) or unrecognized layout."
        )
    cr = profile.counts["catalog_row"]
    fc = profile.counts["file_complete"]
    typer.echo(f"catalog_row events: {cr}  file_complete: {fc}")
    first_cr = profile.first_catalog_row_seconds
    first_fc = profile.first_file_complete_seconds
    if first_cr is not None:
        typer.echo(f"Time to first catalog_row: {format_seconds(first_cr)} s")
    if first_fc is not None:
        typer.echo(f"Time to first file_complete: {format_seconds(first_fc)} s")
    typer.echo()
    typer.echo(render_markdown_table(profile))
    typer.echo(
        "Notes: Python does almost no work during ingest (Rust holds the GIL only for "
        "short callbacks). Slow `headers` on network mounts is mostly FITS open/read "
        "from the volume. `catalog` is single-writer SQLite. `zarr` re-reads each file "
        "for pixels after catalog inserts."
    )
