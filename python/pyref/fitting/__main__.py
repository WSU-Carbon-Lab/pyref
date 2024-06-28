"""

Fitting module for pyref.

This script is the main entry point for the fitting module. When it
is called, it will run the fitting analysis module.
-----------------------------------------------------------------------
Usage:

    python -m pyref.fitting [command] [arguments]

-----------------------------------------------------------------------
Commands:
    struct    Creates a slab structure for the fitting analysis
    model     Creates a model for the fitting analysis
    fit      Fits a model to a dataset
    plot     Plots a model and/or dataset
-----------------------------------------------------------------------

Examples
--------
    python -m pyref.fitting model <structure_file>
    python -m pyref.fitting fit <model_dir> <refl_dir>
    python -m pyref.fitting plot <model_dir> <data_dir>

-----------------------------------------------------------------------
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import _model
import _structure
import art
import click
import rich
import typer
from rich import print

app = typer.Typer(rich_markup_mode="markdown", pretty_exceptions_show_locals=False)

app.add_typer(
    _structure.app,
    name="struct",
    help="Manage slab structures for the fitting analysis",
)
app.add_typer(
    _model.app,
    name="model",
    help="Manage optical constants and models for the fitting analysis",
)

if __name__ == "__main__":
    app()
