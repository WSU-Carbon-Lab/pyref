"""Smoke tests for the Typer CLI."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner


def test_pyref_help() -> None:
    from pyref.cli import app

    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "nas" in r.stdout


def test_catalog_ingest_help() -> None:
    from pyref.cli import app

    runner = CliRunner()
    r = runner.invoke(app, ["catalog", "ingest", "--help"])
    assert r.exit_code == 0
    assert "--max-scans" in r.stdout


def test_ingest_shim_rewrites_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    buf: dict[str, list[str]] = {}

    def fake_app() -> None:
        buf["argv"] = list(sys.argv)

    monkeypatch.setattr("pyref.cli.app", fake_app)
    monkeypatch.setattr(sys, "argv", ["pyref-ingest"])
    from pyref.cli.catalog import ingest_shim_main

    ingest_shim_main(["--beamtime", "/tmp/foo", "--no-progress"])
    assert buf["argv"][1:4] == ["catalog", "ingest", "/tmp/foo"]
