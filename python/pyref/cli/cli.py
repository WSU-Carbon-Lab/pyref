from typing import Optional

import typer
from __init__ import __app__name__, __version__
from init import initialize_refl

app = typer.Typer(add_completion=True)

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app__name__} version {__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True)
) -> None:
    """
    A CLI for the reduction and analysis of reflectometry data.
    """
    return
