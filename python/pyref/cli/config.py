"""
Persistent CLI configuration under the pyref data directory (``config.toml``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w


@dataclass
class PyrefConfig:
    """Single registered NAS root and optional watcher defaults."""

    nas_root: Path | None = None
    watch_default_memory_mb: int = 16
    watch_default_debounce_ms: int = 1500


def _config_path() -> Path:
    from pyref.pyref import py_pyref_data_dir

    return Path(py_pyref_data_dir()).resolve() / "config.toml"


def load() -> PyrefConfig:
    """
    Load ``config.toml`` from the pyref data directory, or return defaults.

    Returns
    -------
    PyrefConfig
        Parsed configuration.
    """
    path = _config_path()
    if not path.is_file():
        return PyrefConfig()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    nas = raw.get("nas") or {}
    watch = raw.get("watch") or {}
    root = nas.get("root")
    return PyrefConfig(
        nas_root=Path(root).resolve() if isinstance(root, str) else None,
        watch_default_memory_mb=int(watch.get("default_memory_mb", 16)),
        watch_default_debounce_ms=int(watch.get("default_debounce_ms", 1500)),
    )


def save(cfg: PyrefConfig) -> None:
    """
    Write ``config.toml`` atomically (tmp + replace).

    Parameters
    ----------
    cfg : PyrefConfig
        Configuration to persist.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "nas": {},
        "watch": {
            "default_memory_mb": cfg.watch_default_memory_mb,
            "default_debounce_ms": cfg.watch_default_debounce_ms,
        },
    }
    if cfg.nas_root is not None:
        data["nas"]["root"] = str(cfg.nas_root.resolve())
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(path)
