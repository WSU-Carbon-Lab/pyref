"""
Beamtime path resolution and environment overrides for the CLI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pyref.cli.config import PyrefConfig, load


def resolve_nas_root(nas_root: Path | None, cfg: PyrefConfig | None = None) -> Path | None:
    """
    Effective NAS root: explicit ``nas_root`` or configured value.

    Parameters
    ----------
    nas_root : pathlib.Path, optional
        Override from a flag.
    cfg : PyrefConfig, optional
        Loaded config; if omitted, :func:`load` is used when ``nas_root`` is None.

    Returns
    -------
    pathlib.Path or None
        Resolved root, or ``None`` if unset.
    """
    if nas_root is not None:
        return nas_root.resolve()
    c = cfg if cfg is not None else load()
    return c.nas_root.resolve() if c.nas_root is not None else None


def resolve_beamtime_path(
    name_or_path: str,
    *,
    nas_root: Path | None = None,
    cfg: PyrefConfig | None = None,
) -> Path:
    """
    Resolve a beamtime directory from an existing path or a name under the NAS root.

    Parameters
    ----------
    name_or_path : str
        Path to a beamtime root that exists on disk, or a single folder name under the NAS root.
    nas_root : pathlib.Path, optional
        Override NAS root for this resolution.
    cfg : PyrefConfig, optional
        Config used when resolving by name.

    Returns
    -------
    pathlib.Path
        Canonical beamtime directory.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or NAS root is missing for a bare name.
    """
    p = Path(name_or_path).expanduser()
    cand = p.resolve()
    if cand.exists():
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"not a directory: {cand}")
    if p.is_absolute() or len(p.parts) != 1:
        raise FileNotFoundError(str(cand))
    root = resolve_nas_root(nas_root, cfg)
    if root is None:
        raise FileNotFoundError(
            "no NAS root configured; use `pyref nas set <PATH>` or `--nas-root`",
        )
    out = (root / name_or_path).resolve()
    if not out.is_dir():
        raise FileNotFoundError(str(out))
    return out


def apply_catalog_env(catalog_db: Path | None, cache_root: Path | None) -> None:
    """
    Set ``PYREF_CATALOG_DB`` and ``PYREF_CACHE_ROOT`` for the current process.

    Parameters
    ----------
    catalog_db : pathlib.Path, optional
        Path to ``catalog.db``.
    cache_root : pathlib.Path, optional
        Parent of ``<hash>/beamtime.zarr`` trees.
    """
    if catalog_db is not None:
        os.environ["PYREF_CATALOG_DB"] = str(catalog_db.resolve())
    if cache_root is not None:
        os.environ["PYREF_CACHE_ROOT"] = str(cache_root.resolve())


def parse_scan_numbers(spec: str | None) -> list[int] | None:
    """
    Parse a comma-separated list of scan numbers.

    Parameters
    ----------
    spec : str, optional
        For example ``"1,3,5"``.

    Returns
    -------
    list of int or None
        Parsed list, or ``None`` if ``spec`` is None/empty.
    """
    if spec is None or not spec.strip():
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            msg = f"invalid scan number token: {p!r}"
            raise ValueError(msg)
        out.append(int(p))
    return out


def daemon_key(beamtime: Path) -> str:
    """
    Stable id for daemon pid/log files from the canonical beamtime path.

    Parameters
    ----------
    beamtime : pathlib.Path
        Beamtime root directory.

    Returns
    -------
    str
        Hex digest string.
    """
    import hashlib

    s = str(beamtime.resolve())
    return hashlib.sha256(s.encode()).hexdigest()
