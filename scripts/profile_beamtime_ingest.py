"""Wall-clock ingest breakdown via Rust progress events (no extra instrumentation).

Run from repo root::

    uv run python scripts/profile_beamtime_ingest.py --beamtime "/path/to/beamtime"

Uses isolated ``PYREF_CATALOG_DB`` and ``PYREF_CACHE_ROOT`` under a temporary directory
unless ``--use-default-paths`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ingest_profile import (
    format_seconds,
    isolated_catalog_env,
    render_markdown_table,
    run_ingest_with_profile,
)


def main() -> None:
    """Print a markdown table of ingest phase wall times for a real beamtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beamtime",
        type=Path,
        required=True,
        help="Beamtime root directory (e.g. ALS date folder containing CCD data).",
    )
    parser.add_argument(
        "--use-default-paths",
        action="store_true",
        help=(
            "Do not override PYREF_CATALOG_DB / PYREF_CACHE_ROOT (writes real catalog)."
        ),
    )
    args = parser.parse_args()
    beam = args.beamtime.resolve()

    with isolated_catalog_env(
        enabled=not args.use_default_paths,
        prefix="pyref-ingest-profile-",
    ):
        _catalog_path, profile = run_ingest_with_profile(beam)

    print(f"Beamtime: {beam}")
    print(f"FITS files (layout): {profile.layout_files}")
    if profile.layout_files == 0:
        print(
            "Note: 0 files often means no ingestible `.fits` "
            "(stems starting with `_` are skipped) or unrecognized layout."
        )
    cr = profile.counts["catalog_row"]
    fc = profile.counts["file_complete"]
    print(f"catalog_row events: {cr}  file_complete: {fc}")
    first_cr = profile.first_catalog_row_seconds
    first_fc = profile.first_file_complete_seconds
    if first_cr is not None:
        print(f"Time to first catalog_row: {format_seconds(first_cr)} s")
    if first_fc is not None:
        print(f"Time to first file_complete: {format_seconds(first_fc)} s")
    print()
    print(render_markdown_table(profile))
    print(
        "Notes: Python does almost no work during ingest (Rust holds the GIL only for "
        "short callbacks). Slow `headers` on network mounts is mostly FITS open/read "
        "from the volume. `catalog` is single-writer SQLite. `zarr` re-reads each file "
        "for pixels after catalog inserts."
    )


if __name__ == "__main__":
    main()
