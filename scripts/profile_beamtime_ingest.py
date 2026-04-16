"""Wall-clock ingest breakdown via Rust progress events (no extra instrumentation).

Run from repo root::

    uv run python scripts/profile_beamtime_ingest.py --beamtime "/path/to/beamtime"

Uses isolated ``PYREF_CATALOG_DB`` and ``PYREF_CACHE_ROOT`` under a temporary directory
unless ``--use-default-paths`` is passed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def _fmt_s(seconds: float) -> str:
    if seconds >= 100.0:
        return f"{seconds:.1f}"
    if seconds >= 10.0:
        return f"{seconds:.2f}"
    return f"{seconds:.3f}"


def main() -> None:
    """Print a markdown table of ingest phase wall times."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--beamtime",
        type=Path,
        required=True,
        help="Beamtime root directory (e.g. ALS date folder containing CCD data).",
    )
    p.add_argument(
        "--use-default-paths",
        action="store_true",
        help=(
            "Do not override PYREF_CATALOG_DB / PYREF_CACHE_ROOT (writes real catalog)."
        ),
    )
    args = p.parse_args()
    beam = args.beamtime.resolve()

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if not args.use_default_paths:
        tmp = tempfile.TemporaryDirectory(prefix="pyref-ingest-profile-")
        tdir = Path(tmp.name)
        os.environ["PYREF_CATALOG_DB"] = str((tdir / "catalog.db").resolve())
        os.environ["PYREF_CACHE_ROOT"] = str((tdir / "cache").resolve())

    from pyref.io.readers import ingest_beamtime

    t0 = time.perf_counter()
    phase_start = t0
    current_phase = "startup"
    phases: dict[str, float] = {}
    counts: dict[str, int] = {
        "layout": 0,
        "catalog_row": 0,
        "file_complete": 0,
    }
    first_fc: float | None = None
    first_catalog: float | None = None
    layout_files = 0

    def on_progress(d: dict[str, Any]) -> None:
        nonlocal phase_start, current_phase, first_fc, first_catalog, layout_files
        ev = d.get("event")
        now = time.perf_counter()
        if ev == "phase":
            n = str(d.get("phase", ""))
            phases[current_phase] = phases.get(current_phase, 0.0) + (now - phase_start)
            current_phase = n
            phase_start = now
            return
        if ev == "layout":
            layout_files = int(d.get("total_files", 0))
            counts["layout"] += 1
            return
        if ev == "catalog_row":
            if first_catalog is None:
                first_catalog = now - t0
            counts["catalog_row"] += 1
            return
        if ev == "file_complete":
            if first_fc is None:
                first_fc = now - t0
            counts["file_complete"] += 1

    try:
        ingest_beamtime(beam, None, progress_callback=on_progress)
    finally:
        if tmp is not None:
            tmp.cleanup()

    wall = time.perf_counter() - t0
    tail = time.perf_counter() - phase_start
    phases[current_phase] = phases.get(current_phase, 0.0) + tail

    startup_s = phases.get("startup", 0.0)
    headers_s = phases.get("headers", 0.0)
    catalog_s = phases.get("catalog", 0.0)
    zarr_s = phases.get("zarr", 0.0)
    accounted = startup_s + headers_s + catalog_s + zarr_s
    other_s = max(0.0, wall - accounted)

    rows = [
        ("Before `headers` (DB open, discovery, layout)", startup_s),
        ("Phase `headers` (parallel FITS header reads)", headers_s),
        ("Phase `catalog` (SQLite transaction + `catalog_row` events)", catalog_s),
        ("Phase `zarr` (FITS pixels read + zarr write + `file_complete`)", zarr_s),
        ("Unattributed (measurement gap)", other_s),
        ("Wall time total", wall),
    ]

    print(f"Beamtime: {beam}")
    print(f"FITS files (layout): {layout_files}")
    if layout_files == 0:
        print(
            "Note: 0 files often means no ingestible `.fits` "
            "(stems starting with `_` are skipped) or unrecognized layout."
        )
    cr, fc = counts["catalog_row"], counts["file_complete"]
    print(f"catalog_row events: {cr}  file_complete: {fc}")
    if first_catalog is not None:
        print(f"Time to first catalog_row: {_fmt_s(first_catalog)} s")
    if first_fc is not None:
        print(f"Time to first file_complete: {_fmt_s(first_fc)} s")
    print()
    print("| Segment | Seconds | Share of wall |")
    print("|---------|--------:|--------------:|")
    for label, sec in rows[:-1]:
        share = (sec / wall * 100.0) if wall > 0 else 0.0
        print(f"| {label} | {_fmt_s(sec)} | {share:.1f}% |")
    label, sec = rows[-1]
    print(f"| **{label}** | **{_fmt_s(sec)}** | **100%** |")
    print()
    print(
        "Notes: Python does almost no work during ingest (Rust holds the GIL only for "
        "short callbacks). Slow `headers` on network mounts is mostly FITS open/read "
        "from the volume. `catalog` is single-writer SQLite. `zarr` re-reads each file "
        "for pixels after catalog inserts."
    )


if __name__ == "__main__":
    main()
