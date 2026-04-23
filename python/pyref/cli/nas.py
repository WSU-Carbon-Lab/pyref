"""
``pyref nas``: register the single NAS root used to resolve beamtime folder names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from pyref.cli.config import load, save

app = typer.Typer(help="Registered NAS root for beamtime name resolution")


@app.command("set")
def nas_set(path: Path) -> None:
    """
    Persist the absolute path to the NAS (or data) root for future ``NAME`` lookups.
    """
    p = path.expanduser().resolve()
    if not p.is_dir():
        sys.stderr.write(f"error: not a directory: {p}\n")
        raise typer.Exit(1)
    cfg = load()
    cfg.nas_root = p
    save(cfg)
    typer.echo(str(p))


@app.command("show")
def nas_show() -> None:
    """Print the registered NAS root and whether it exists."""
    cfg = load()
    if cfg.nas_root is None:
        typer.echo("(no NAS root registered)")
        raise typer.Exit(0)
    root = cfg.nas_root.resolve()
    typer.echo(f"root: {root}")
    typer.echo(f"exists: {root.is_dir()}")


@app.command("clear")
def nas_clear() -> None:
    """Remove the stored NAS root from config."""
    cfg = load()
    cfg.nas_root = None
    save(cfg)
