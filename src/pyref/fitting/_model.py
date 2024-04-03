import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import art
import rich
import typer
from rich import print
from typing_extensions import Annotated

app = typer.Typer(rich_markup_mode="markdown", pretty_exceptions_show_locals=False)

dbapp = typer.Typer(rich_markup_mode="markdown", pretty_exceptions_show_locals=False)

@dbapp.command()
def add(
    name: str = typer.Argument(
        ..., 
        help="Material name, or chmical formula if nexafs is not provided",
        show_default=False,
    ),
    nexafs: Path = typer.Option(
        None,
        "--nexafs", 
        "-n", 
        help="Path to the NEXAFS file", 
        exists=True,
        show_default=False,
        rich_help_panel="Optical constant source"
    ),
    bare_atom: str = typer.Option(
        None,
        "--bare-atom",
        "-b", 
        help="Path to the NEXAFS file", 
        exists=True,
        show_default=False,
        rich_help_panel="Optical constant source",
    ),
    description: str = typer.Option(
        None, "--description", "-d", help="Description of the optical constant",
        show_default=False,
    ),
    overwrite: bool = typer.Option(
        False, 
        "--overwrite", 
        "-o", 
        help="Overwrite existing optical constant",
    ),
    symmetry: str = typer.Option(
        "uni", 
        "--symmetry", 
        "-s", 
        help="Symmetry of the material - 'iso', 'uni', 'bi'",
        rich_help_panel="Material properties"
    ),
    energy_max: float = typer.Option(
        "320", 
        "--max-energy",
        "-max", 
        help="Maximum energy of the optical constant [eV]",
        rich_help_panel="Material properties"
    ),
    energy_min: float = typer.Option(
        "250", 
        "--min-energy", 
        "-min",
        help="Minimum energy of the optical constant [eV]",
        rich_help_panel="Material properties"
    ),
):
    """
    # **Add** a material to the database :heavy_plus_sign:

    ---

    * If nexafs is not provided, the optical constant will try to be evaluated using the name as a chemical formula.
    """

    ...

@dbapp.command()
def remove(
    name: str = typer.Argument(..., help="Chemical name of the optical constant")
):
    """
    **Remove** a material from the database :heavy_minus_sign:
    """
    ...

@dbapp.command()
def list():
    """
    **List** all materials in the database :page_facing_up:
    """
    ...

@dbapp.command()
def info(
    name: str = typer.Argument(..., help="Chemical name of the optical constant")
):
    """
    **Info** about a material in the database :mag:
    """
    ...



app.add_typer(dbapp, name="db", help="Manage optical constant database")