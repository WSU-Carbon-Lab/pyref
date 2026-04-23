"""
``pyref beamtime``: list and describe beamtimes without ingesting pixels unnecessarily.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from pyref.cli.config import load as load_config
from pyref.cli.resolve import resolve_beamtime_path
from pyref.io.readers import beamtime_ingest_layout

app = typer.Typer(help="Beamtime discovery and coverage")


@app.command("list")
def beamtime_list(
    cataloged: bool = typer.Option(
        False,
        "--cataloged",
        help="Only beamtimes whose catalog file count matches on-disk FITS count.",
    ),
    catalog_db: Path | None = typer.Option(
        None,
        "--catalog-db",
        help="Override PYREF_CATALOG_DB for this command.",
    ),
) -> None:
    """List beamtimes recorded in the catalog (most recent first)."""
    from pyref.io.catalog_path import resolve_catalog_path
    from pyref.pyref import py_catalog_file_count, py_list_beamtimes

    db = catalog_db.resolve() if catalog_db is not None else resolve_catalog_path()
    if not db.is_file():
        sys.stderr.write(f"error: catalog database not found: {db}\n")
        raise typer.Exit(1)
    rows = py_list_beamtimes(str(db))
    for path_str, _bid in rows:
        p = Path(path_str)
        if cataloged:
            if not p.is_dir():
                continue
            try:
                layout = beamtime_ingest_layout(p)
            except OSError:
                continue
            total = int(layout["total_files"])
            if total == 0:
                continue
            n = py_catalog_file_count(str(db), str(p.resolve()))
            if n < total:
                continue
        typer.echo(f"{path_str}")


@app.command("describe")
def beamtime_describe(
    name: str = typer.Argument(..., help="Beamtime folder name or absolute path."),
    nas_root: Path | None = typer.Option(
        None,
        "--nas-root",
        help="Override the configured NAS root for name resolution.",
    ),
    catalog_db: Path | None = typer.Option(
        None,
        "--catalog-db",
        help="Override PYREF_CATALOG_DB.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """
    Print scan count, FITS on disk, cataloged count, and coverage percent.
    """
    from pyref.io.catalog_path import resolve_catalog_path
    from pyref.pyref import py_catalog_file_count

    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc

    db = catalog_db.resolve() if catalog_db is not None else resolve_catalog_path()
    layout = beamtime_ingest_layout(bt)
    disk_files = int(layout["total_files"])
    n_scans = len(layout["scans"])
    n_cat = 0
    if db.is_file():
        try:
            n_cat = py_catalog_file_count(str(db), str(bt.resolve()))
        except OSError as exc:
            sys.stderr.write(f"error: catalog query failed: {exc}\n")
            raise typer.Exit(2) from exc
    pct = (100.0 * n_cat / disk_files) if disk_files else 0.0
    payload = {
        "beamtime": str(bt.resolve()),
        "scans_on_disk": n_scans,
        "fits_files_on_disk": disk_files,
        "fits_files_in_catalog": n_cat,
        "percent_cataloged": round(pct, 3),
    }
    if as_json:
        typer.echo(json.dumps(payload))
    else:
        typer.echo(f"beamtime: {payload['beamtime']}")
        typer.echo(f"scans (layout): {n_scans}")
        typer.echo(f"fits on disk: {disk_files}")
        typer.echo(f"fits in catalog: {n_cat}")
        typer.echo(f"percent cataloged: {pct:.2f}%")
