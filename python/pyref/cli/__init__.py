"""
Typer CLI: NAS registration, beamtime describe, catalog ingest, watch, bench.
"""

from __future__ import annotations

import typer

from pyref.cli import beamtime as beamtime_cli
from pyref.cli import bench as bench_cli
from pyref.cli import catalog as catalog_cli
from pyref.cli import nas as nas_cli
from pyref.cli import watch as watch_cli

app = typer.Typer(
    help="pyref: beamtime catalog and incremental ingest",
    no_args_is_help=True,
)
app.add_typer(nas_cli.app, name="nas")
app.add_typer(beamtime_cli.app, name="beamtime")
app.add_typer(catalog_cli.app, name="catalog")
app.add_typer(watch_cli.app, name="watch")
app.add_typer(bench_cli.app, name="bench")

__all__ = ["app"]
