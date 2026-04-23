"""
Console entry points registered in ``pyproject.toml``.
"""

from __future__ import annotations


def main() -> None:
    """Run the ``pyref`` Typer application."""
    from pyref.cli import app

    app()


def main_ingest_shim() -> None:
    """Legacy ``pyref-ingest`` entry: map ``--beamtime`` to ``catalog ingest``."""
    from pyref.cli.catalog import ingest_shim_main

    ingest_shim_main()
