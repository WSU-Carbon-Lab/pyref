"""
Resolve the global SQLite catalog path.

The catalog is a single database shared across beamtimes. Its default location is
``$XDG_CONFIG_HOME/pyref/catalog.db`` when ``XDG_CONFIG_HOME`` is set, otherwise
``~/.config/pyref/catalog.db`` (Unix-like under the user home on all platforms). When
``PYREF_HOME`` is set, the default becomes ``<PYREF_HOME>/catalog.db``. The beamtime zarr
cache remains under the platform user data directory (see Rust ``pyref_data_dir``), not
next to this config path.

Optional overrides (Rust IO layer): ``PYREF_CATALOG_DB`` forces the catalog file path;
``PYREF_CACHE_ROOT`` sets the parent directory of each ``<beamtime_hash>/beamtime.zarr``
tree (default remains ``<pyref_data_dir>/.cache/``). Parallel ingest reads
``PYREF_INGEST_WORKER_THREADS`` or ``PYREF_INGEST_RESOURCE_FRACTION`` when Python
passes neither ``worker_threads`` nor ``resource_fraction`` to ``ingest_beamtime``.
"""

from __future__ import annotations

from pathlib import Path

NEW_CATALOG_DB_NAME = "catalog.db"
PYREF_CATALOG_DIR = ".pyref"


def catalog_path_new(_beamtime_dir: Path | None = None) -> Path:
    """
    Return the default catalog database path (global; beamtime argument ignored).

    Parameters
    ----------
    _beamtime_dir : pathlib.Path, optional
        Ignored. Retained for compatibility with older signatures.

    Returns
    -------
    pathlib.Path
        Path to ``catalog.db`` (may not exist yet).
    """
    return resolve_catalog_path()


def resolve_catalog_path(_beamtime_dir: str | Path | None = None) -> Path:
    """
    Return the default catalog database path via the Rust extension.

    Parameters
    ----------
    _beamtime_dir : str or pathlib.Path, optional
        Ignored. Retained for compatibility with older signatures.

    Returns
    -------
    pathlib.Path
        Resolved absolute path to the catalog SQLite file.
    """
    from pyref.pyref import py_default_catalog_db_path

    _ = _beamtime_dir
    return Path(py_default_catalog_db_path()).resolve()
