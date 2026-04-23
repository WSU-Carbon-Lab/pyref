"""
``pyref catalog``: show catalog paths and ingest beamtimes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import typer

from pyref.cli.config import load as load_config
from pyref.cli.resolve import (
    apply_catalog_env,
    parse_scan_numbers,
    resolve_beamtime_path,
)
from pyref.io.beamtime import ingest_beamtime_with_rich_progress
from pyref.io.catalog_path import resolve_catalog_path
from pyref.io.experiment_names import discover_fits, parse_fits_stem
from pyref.io.readers import (
    DEFAULT_HEADER_KEYS,
    beamtime_ingest_layout,
    ingest_beamtime,
)

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
    from pyref.pyref import py_pyref_data_dir

    db = catalog_db.resolve() if catalog_db is not None else resolve_catalog_path()
    cache = os.environ.get("PYREF_CACHE_ROOT")
    if cache is None:
        cache = str(Path(py_pyref_data_dir()).resolve() / "cache")
    typer.echo(f"catalog_db: {db}")
    typer.echo(f"cache_root: {cache}")


def _scan_counts_from_catalog(df: pl.DataFrame) -> dict[int, int]:
    if df.height == 0:
        return {}
    grouped = (
        df.group_by("scan_number")
        .agg(pl.len().alias("n"))
        .sort("scan_number")
        .to_dicts()
    )
    return {int(row["scan_number"]): int(row["n"]) for row in grouped}


def _disk_frame_keys(beamtime_path: Path) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for path in discover_fits(beamtime_path, recursive=True):
        parsed = parse_fits_stem(path.stem)
        if parsed is None:
            continue
        out.add((int(parsed.scan_number), int(parsed.frame_number)))
    return out


def _catalog_frame_keys(df: pl.DataFrame) -> set[tuple[int, int]]:
    if df.height == 0:
        return set()
    rows = df.select("scan_number", "frame_number").to_dicts()
    return {(int(r["scan_number"]), int(r["frame_number"])) for r in rows}


def _resolve_sync_targets(
    *,
    name: str | None,
    latest: bool,
    all_beamtimes: bool,
    nas_root: Path | None,
) -> list[Path]:
    from pyref.pyref import py_list_beamtimes

    selector_count = int(name is not None) + int(latest) + int(all_beamtimes)
    if selector_count != 1:
        msg = "select exactly one target: <beamtime_name>, --latest, or --all"
        raise ValueError(msg)
    cfg = load_config()
    if name is not None:
        return [resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)]
    db = resolve_catalog_path()
    rows = py_list_beamtimes(str(db))
    paths = [Path(path_str).resolve() for path_str, _ in rows if Path(path_str).is_dir()]
    if latest:
        if not paths:
            msg = f"no cataloged beamtimes found in {db}"
            raise FileNotFoundError(msg)
        return [paths[0]]
    return paths


@app.command("sync")
def catalog_sync(
    name: str | None = typer.Argument(
        None,
        help="Beamtime folder name or path (mutually exclusive with --latest/--all).",
    ),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="Sync the most recently cataloged beamtime.",
    ),
    all_beamtimes: bool = typer.Option(
        False,
        "--all",
        help="Sync every cataloged beamtime available on disk.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help="Backfill missing per-file header rows in header_values.",
    ),
    nas_root: Path | None = typer.Option(
        None,
        "--nas-root",
        help="Override configured NAS root when resolving a beamtime name.",
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
) -> None:
    """Audit catalog coverage per beamtime; optionally backfill missing header rows."""
    from pyref.pyref import (
        py_scan_from_catalog_for_beamtime,
        py_sync_missing_headers_for_beamtime,
    )

    apply_catalog_env(catalog_db, cache_root)
    db = catalog_db.resolve() if catalog_db is not None else resolve_catalog_path()
    try:
        targets = _resolve_sync_targets(
            name=name,
            latest=latest,
            all_beamtimes=all_beamtimes,
            nas_root=nas_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc

    if not targets:
        typer.echo("no beamtime targets found")
        return

    for beamtime in targets:
        layout = beamtime_ingest_layout(beamtime)
        layout_scan_counts = {
            int(row["scan_number"]): int(row["files"]) for row in layout.get("scans", [])
        }
        disk_total = int(layout.get("total_files", 0))
        catalog_df = py_scan_from_catalog_for_beamtime(str(db), str(beamtime), None)
        cat_scan_counts = _scan_counts_from_catalog(catalog_df)
        cat_total = int(catalog_df.height)

        missing_scans = sorted(set(layout_scan_counts) - set(cat_scan_counts))
        missing_files_by_scan = {
            scan: layout_scan_counts[scan] - cat_scan_counts.get(scan, 0)
            for scan in sorted(layout_scan_counts)
            if layout_scan_counts[scan] - cat_scan_counts.get(scan, 0) > 0
        }
        disk_keys = _disk_frame_keys(beamtime)
        cat_keys = _catalog_frame_keys(catalog_df)
        missing_keys = sorted(disk_keys - cat_keys)

        typer.echo(f"beamtime: {beamtime}")
        typer.echo(f"  scans on disk: {len(layout_scan_counts)}")
        typer.echo(f"  scans in catalog: {len(cat_scan_counts)}")
        if missing_scans:
            typer.echo(f"  missing scans: {', '.join(str(s) for s in missing_scans)}")
        else:
            typer.echo("  missing scans: none")
        typer.echo(f"  files on disk: {disk_total}")
        typer.echo(f"  files in catalog: {cat_total}")
        typer.echo(f"  missing files: {len(missing_keys)}")
        if missing_files_by_scan:
            summary = ", ".join(
                f"{scan}:{count}" for scan, count in missing_files_by_scan.items()
            )
            typer.echo(f"  missing files by scan: {summary}")

        if update:
            report = py_sync_missing_headers_for_beamtime(str(db), str(beamtime))
            typer.echo(
                "  header sync: "
                f"checked={report['files_checked']} "
                f"updated={report['files_updated']} "
                f"inserted={report['header_rows_inserted']}",
            )


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
