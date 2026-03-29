"""
Resolve SQLite catalog paths for a beamtime directory.

Mirrors ``resolve_catalog_path`` and related helpers in the Rust catalog module
(``src/catalog/mod.rs``). The new layout stores a shared database at
``parent/.pyref/catalog.db``; the legacy layout uses ``beamtime_dir/.pyref_catalog.db``.
Existence checks use :meth:`pathlib.Path.exists` (symlinks are followed), matching
Rust ``exists()``.
"""

from __future__ import annotations

from pathlib import Path

CATALOG_DB_NAME = ".pyref_catalog.db"
PYREF_CATALOG_DIR = ".pyref"
NEW_CATALOG_DB_NAME = "catalog.db"


def catalog_path_legacy(beamtime_dir: Path) -> Path:
    """
    Return the legacy per-beamtime catalog database path.

    Parameters
    ----------
    beamtime_dir : pathlib.Path
        Beamtime root directory. Not required to exist.

    Returns
    -------
    pathlib.Path
        ``beamtime_dir / CATALOG_DB_NAME`` (``CATALOG_DB_NAME`` is
        ``.pyref_catalog.db``).
    """
    return beamtime_dir / CATALOG_DB_NAME


def catalog_path_new(beamtime_dir: Path) -> Path:
    """
    Return the new-layout catalog database path for a beamtime directory.

    If ``beamtime_dir`` has a parent that is non-empty and not the filesystem
    root, the path is ``parent / .pyref / catalog.db``. Otherwise it is
    ``beamtime_dir / .pyref / catalog.db``, matching the Rust ``catalog_path_new``.

    Parameters
    ----------
    beamtime_dir : pathlib.Path
        Beamtime root directory. Not required to exist.

    Returns
    -------
    pathlib.Path
        Candidate path for the shared catalog in the new layout.
    """
    parent = beamtime_dir.parent
    ps = parent.as_posix()
    if ps and ps != "/":
        return parent / PYREF_CATALOG_DIR / NEW_CATALOG_DB_NAME
    return beamtime_dir / PYREF_CATALOG_DIR / NEW_CATALOG_DB_NAME


def resolve_catalog_path(beamtime_dir: str | Path) -> Path:
    """
    Choose the catalog database path for a beamtime directory.

    Resolves ``beamtime_dir`` with :meth:`pathlib.Path.resolve` for stable,
    absolute paths. If the new-layout path exists, returns it; else if the
    legacy path exists, returns that; else returns the new-layout path (the
    default location ingest would create in the new layout).

    Parameters
    ----------
    beamtime_dir : str or pathlib.Path
        Beamtime root directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the catalog database file to use.
    """
    d = Path(beamtime_dir).resolve()
    new_path = catalog_path_new(d)
    legacy_path = catalog_path_legacy(d)
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path
