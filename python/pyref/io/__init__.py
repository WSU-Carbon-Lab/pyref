"""
I/O Operations for Reflectivity Data at BL 11.0.1.2.

This module provides a collection of tools for reading and writing data related to
X-ray reflectivity experiments, particularly those conducted at beamline 11.0.1.2.
The primary focus is on handling FITS files, which are commonly used for storing
astronomical and scientific image data.

The functions aim to provide a simple and consistent interface for loading experimental
data into standard data analysis structures like pandas or polars DataFrames.

Input/Output operations currently supported:
===========================================

FITS Files
----------
- :func:`~pyref.io.read_fits`: Read data from one or more FITS files.
- :func:`~pyref.io.read_experiment`: Read all FITS files from a directory or based on a
        pattern.

Experiment catalog and discovery
-------------------------------
- :func:`~pyref.io.read_beamtime`: Ingest (optional) and load one beamtime's samples,
  scans, and frames.
- :func:`~pyref.io.list_beamtimes`: Enumerate beamtime roots stored in the catalog.
- :func:`~pyref.io.resolve_catalog_path`: Beamtime catalog database path (Rust-aligned).
- :func:`~pyref.io.classify_reflectivity_scan_type`: Theta vs energy scan
  classification.
- :func:`~pyref.io.discover_fits`: Discover FITS files under a root path (flat or
  recursive).
- :func:`~pyref.io.parse_fits_stem`: Parse a FITS filename stem into sample_name, tag,
  scan_number, frame_number.
- :func:`~pyref.io.build_catalog`: Build a per-file catalog DataFrame (names-only or
  with headers).
- :func:`~pyref.io.scan_view`: Aggregate catalog into a per-scan view (file_count,
  energy/Q range).
- :func:`~pyref.io.experiment_summary`: Quick view of an experiment directory (scan
  summary table).
- :func:`~pyref.io.filter_catalog_paths`: Filter catalog by sample_name, tag, or scan
  number(s).

See the specific function documentation for more details on usage and parameters.
"""

from pyref.io import fits_accessor  # noqa: F401 - registers df.fits accessor
from pyref.io.beamtime import (
    BeamtimeCatalogView,
    BeamtimeEntriesView,
    beamtime_entries,
    ingest_beamtime_with_rich_progress,
    list_beamtimes,
    naming_qc_from_frames,
    naming_qc_with_db_parse_flags,
    read_beamtime,
    scan_from_catalog_for_beamtime,
)
from pyref.io.catalog_path import resolve_catalog_path
from pyref.io.experiment_names import (
    ParsedFitsName,
    build_catalog,
    discover_fits,
    experiment_summary,
    filter_catalog_paths,
    parse_fits_stem,
    scan_view,
)
from pyref.io.readers import (
    beamtime_ingest_layout,
    classify_reflectivity_scan_type,
    get_image,
    get_image_corrected,
    get_image_filtered,
    get_image_filtered_edges,
    get_overrides,
    ingest_beamtime,
    query_catalog,
    read_experiment,
    read_fits,
    resolve_fits_paths,
    scan_experiment,
    set_override,
)

__all__ = [
    "BeamtimeCatalogView",
    "BeamtimeEntriesView",
    "ParsedFitsName",
    "beamtime_entries",
    "beamtime_ingest_layout",
    "build_catalog",
    "classify_reflectivity_scan_type",
    "discover_fits",
    "experiment_summary",
    "filter_catalog_paths",
    "get_image",
    "get_image_corrected",
    "get_image_filtered",
    "get_image_filtered_edges",
    "get_overrides",
    "ingest_beamtime",
    "ingest_beamtime_with_rich_progress",
    "list_beamtimes",
    "naming_qc_from_frames",
    "naming_qc_with_db_parse_flags",
    "parse_fits_stem",
    "query_catalog",
    "read_beamtime",
    "read_experiment",
    "read_fits",
    "resolve_catalog_path",
    "resolve_fits_paths",
    "scan_experiment",
    "scan_from_catalog_for_beamtime",
    "scan_view",
    "set_override",
]
