"""
``pyref watch``: long-running file watcher daemon for incremental ingest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from pyref.cli.config import load as load_config
from pyref.cli.daemon import (
    list_records,
    start_watch_daemon_subprocess,
    stop_daemon,
    tail_log,
)
from pyref.cli.resolve import daemon_key, resolve_beamtime_path
from pyref.io.readers import DEFAULT_HEADER_KEYS

app = typer.Typer(help="Watch beamtime folders and ingest new FITS files")


@app.command("start")
def watch_start(
    name: str = typer.Argument(..., help="Beamtime folder name or path."),
    nas_root: Path | None = typer.Option(None, "--nas-root", help="Override NAS root."),
    subpath: list[str] = typer.Option(
        [],
        "--subpath",
        help="Relative subfolder to watch (repeatable). Default: full beamtime tree.",
    ),
    debounce_ms: int = typer.Option(1500, "--debounce-ms", help="Debounce interval in ms."),
    memory_mb: int = typer.Option(16, "--memory-mb", help="Soft memory budget (see docs)."),
    cpu: int | None = typer.Option(None, "--cpu", help="Linux: pin worker to this CPU id."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Log file path."),
    workers: int = typer.Option(1, "--workers", help="Ingest worker threads per triggered run."),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run in the foreground (no subprocess).",
    ),
    header: list[str] | None = typer.Option(
        None,
        "--header",
        help="FITS header key; repeatable.",
    ),
) -> None:
    """Start a watcher daemon for one beamtime."""
    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc
    keys = list(header) if header else list(DEFAULT_HEADER_KEYS)
    if foreground:
        import os

        os.environ["PYREF_INGEST_WORKER_THREADS"] = str(max(1, workers))
        from pyref.pyref import CatalogWatcherCancel, py_run_catalog_watcher_blocking

        cancel = CatalogWatcherCancel()

        def handle_sig(_sig: int, _frame: object) -> None:
            cancel.cancel()

        import signal

        signal.signal(signal.SIGTERM, handle_sig)
        signal.signal(signal.SIGINT, handle_sig)
        try:
            py_run_catalog_watcher_blocking(
                str(bt.resolve()),
                keys,
                debounce_ms,
                list(subpath),
                cancel,
            )
        except OSError as exc:
            sys.stderr.write(f"error: watcher failed: {exc}\n")
            raise typer.Exit(3) from exc
        return
    pid = start_watch_daemon_subprocess(
        bt,
        keys,
        debounce_ms,
        list(subpath),
        memory_mb,
        cpu,
        workers,
        log_file,
    )
    typer.echo(str(pid))


@app.command("stop")
def watch_stop(
    name: str = typer.Argument(..., help="Beamtime folder name or path."),
    nas_root: Path | None = typer.Option(None, "--nas-root"),
) -> None:
    """Stop the watcher for this beamtime."""
    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc
    stop_daemon(bt)


@app.command("status")
def watch_status(
    name: str | None = typer.Argument(
        None,
        help="Optional beamtime; when omitted, list all recorded daemons.",
    ),
    nas_root: Path | None = typer.Option(None, "--nas-root"),
) -> None:
    """Show watcher PID and metadata."""
    recs = list_records()
    if name is None:
        for _k, r in sorted(recs.items(), key=lambda x: x[1].started_at, reverse=True):
            typer.echo(
                f"pid={r.pid} beamtime={r.beamtime} subpaths={r.subpaths} "
                f"started={r.started_at} log={r.log_file}",
            )
        return
    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc
    key = daemon_key(bt)
    r = recs.get(key)
    if r is None:
        typer.echo("(no daemon record)")
        return
    typer.echo(f"pid={r.pid} beamtime={r.beamtime} log={r.log_file}")


@app.command("logs")
def watch_logs(
    name: str = typer.Argument(..., help="Beamtime folder name or path."),
    nas_root: Path | None = typer.Option(None, "--nas-root"),
    tail: int = typer.Option(80, "--tail", "-n", help="Last N lines."),
    follow: bool = typer.Option(False, "--follow", "-f"),
) -> None:
    """Print watcher log output."""
    cfg = load_config()
    try:
        bt = resolve_beamtime_path(name, nas_root=nas_root, cfg=cfg)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise typer.Exit(1) from exc
    key = daemon_key(bt)
    r = list_records().get(key)
    log_path = Path(r.log_file) if r is not None else Path()
    if r is None:
        from pyref.pyref import py_pyref_data_dir

        log_path = Path(py_pyref_data_dir()).resolve() / "daemons" / f"{key}.log"
    tail_log(log_path, tail, follow)


@app.command("list")
def watch_list() -> None:
    """List recorded watcher processes (same as ``watch status`` with no beamtime)."""
    recs = list_records()
    for _k, r in sorted(recs.items(), key=lambda x: x[1].started_at, reverse=True):
        typer.echo(
            f"pid={r.pid} beamtime={r.beamtime} subpaths={r.subpaths} "
            f"started={r.started_at} log={r.log_file}",
        )
