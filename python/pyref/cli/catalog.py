"""
``pyref catalog``: show catalog paths and ingest beamtimes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from pyref.cli.config import load as load_config
from pyref.cli.resolve import (
    apply_catalog_env,
    parse_scan_numbers,
    resolve_beamtime_path,
)
from pyref.io.beamtime import ingest_beamtime_with_rich_progress
from pyref.io.readers import DEFAULT_HEADER_KEYS, ingest_beamtime

app = typer.Typer(help="Global catalog and ingest")


@app.command("path")
def catalog_path_cmd(
    catalog_db: Path | None = typer.Option(
        None,
        "--catalog-db",
        help="Show this path instead of the resolved default.",
    ),
) -> None:
    """Print the active catalog database and zarr cache root."""
    from pyref.io.catalog_path import resolve_catalog_path
    from pyref.pyref import py_pyref_data_dir

    db = catalog_db.resolve() if catalog_db is not None else resolve_catalog_path()
    cache = os.environ.get("PYREF_CACHE_ROOT")
    if cache is None:
        cache = str(Path(py_pyref_data_dir()).resolve() / ".cache")
    typer.echo(f"catalog_db: {db}")
    typer.echo(f"cache_root: {cache}")


@app.command("ingest")
def catalog_ingest(
    name: str = typer.Argument(..., help="Beamtime folder name or path."),
    nas_root: Path | None = typer.Option(
        None,
        "--nas-root",
        help="Override configured NAS root.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        help="Parallel FITS reader worker count.",
    ),
    resource_fraction: float | None = typer.Option(
        None,
        "--resource-fraction",
        help="Fraction of CPUs for reader workers (0,1]; exclusive with --workers.",
    ),
    max_scans: int | None = typer.Option(
        None,
        "--max-scans",
        help="Ingest only the first N scans (ascending scan number).",
    ),
    scans: str | None = typer.Option(
        None,
        "--scans",
        help="Comma-separated scan numbers (e.g. 1,3,5). Exclusive with --max-scans.",
    ),
    header: list[str] | None = typer.Option(
        None,
        "--header",
        help="FITS header key; repeatable. Defaults to built-in list when omitted.",
    ),
    catalog_db: Path | None = typer.Option(
        None,
        "--catalog-db",
        help="Set PYREF_CATALOG_DB for this run.",
    ),
    cache_root: Path | None = typer.Option(
        None,
        "--cache-root",
        help="Set PYREF_CACHE_ROOT for this run.",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable Rich progress (CI / logs).",
    ),
) -> None:
    """Ingest one beamtime into the SQLite catalog and local zarr cache."""
    if workers is not None and resource_fraction is not None:
        sys.stderr.write("error: use only one of --workers or --resource-fraction\n")
        raise typer.Exit(1)
    if max_scans is not None and scans is not None:
        sys.stderr.write("error: use only one of --max-scans or --scans\n")
        raise typer.Exit(1)
    try:
        scan_numbers = parse_scan_numbers(scans)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc
    if resource_fraction is not None and not (0.0 < resource_fraction <= 1.0):
        sys.stderr.write("error: --resource-fraction must be in (0, 1]\n")
        raise typer.Exit(1)

    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc

    apply_catalog_env(catalog_db, cache_root)
    keys = list(header) if header else list(DEFAULT_HEADER_KEYS)
    try:
        if no_progress:
            out = ingest_beamtime(
                bt,
                keys,
                incremental=True,
                worker_threads=workers,
                resource_fraction=resource_fraction,
                max_scans=max_scans,
                scan_numbers=scan_numbers,
            )
        else:
            out = ingest_beamtime_with_rich_progress(
                bt,
                keys,
                worker_threads=workers,
                resource_fraction=resource_fraction,
                max_scans=max_scans,
                scan_numbers=scan_numbers,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"error: ingest failed: {exc}\n")
        raise typer.Exit(2) from exc
    typer.echo(f"catalog: {out}")


def ingest_shim_main(argv: list[str] | None = None) -> None:
    """
    ``pyref-ingest`` compatibility: map legacy ``--beamtime`` to positional argument.
    """
    a = list(argv if argv is not None else sys.argv[1:])
    out: list[str] = ["catalog", "ingest"]
    if "--beamtime" in a:
        i = a.index("--beamtime")
        try:
            bt = a[i + 1]
        except IndexError:
            sys.stderr.write("error: --beamtime requires a value\n")
            raise SystemExit(1) from None
        a = a[:i] + a[i + 2 :]
        out.append(bt)
    elif a and not a[0].startswith("-"):
        out.append(a.pop(0))
    sys.argv = [sys.argv[0], *out, *a]
    from pyref.cli import app

    app()
