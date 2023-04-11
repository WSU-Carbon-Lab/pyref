"""Console script for prsoxs_xrr."""
import sys
import click


@click.command()
def main(args=None) -> Literal[0]:
    """Console script for prsoxs_xrr."""
    click.echo("Replace this message by putting your code into "
               "prsoxs_xrr.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
