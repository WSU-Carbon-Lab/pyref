"""
Resolve SQLite catalog paths for a beamtime directory.

Mirrors ``resolve_catalog_path`` in the Rust catalog module (``src/catalog/mod.rs``).
The catalog is stored at ``parent/.pyref/catalog.db`` when ``beamtime_dir`` has a
non-root parent; otherwise at ``beamtime_dir/.pyref/catalog.db``.
"""

from __future__ import annotations

from pathlib import Path

PYREF_CATALOG_DIR = ".pyref"
NEW_CATALOG_DB_NAME = "catalog.db"


def catalog_path_new(beamtime_dir: Path) -> Path:
    """
    Return the catalog database path for a beamtime directory.

    Parameters
    ----------
    beamtime_dir : pathlib.Path
        Beamtime root directory. Not required to exist.

    Returns
    -------
    pathlib.Path
        ``parent / .pyref / catalog.db`` when ``beamtime_dir`` has a usable parent,
        else ``beamtime_dir / .pyref / catalog.db``.
    """
    parent = beamtime_dir.parent
    ps = parent.as_posix()
    if ps and ps != "/":
        return parent / PYREF_CATALOG_DIR / NEW_CATALOG_DB_NAME
    return beamtime_dir / PYREF_CATALOG_DIR / NEW_CATALOG_DB_NAME


def resolve_catalog_path(beamtime_dir: str | Path) -> Path:
    """
    Return the catalog database path for a beamtime directory.

    Resolves ``beamtime_dir`` with :meth:`pathlib.Path.resolve` for stable paths.

    Parameters
    ----------
    beamtime_dir : str or pathlib.Path
        Beamtime root directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the catalog database file (may not exist yet).
    """
    d = Path(beamtime_dir).resolve()
    return catalog_path_new(d).resolve()
