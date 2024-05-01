""" 
    pyref init command
    ==================
    This command initializes the database within the current folder, 
    This will create a .refl folder within the current directory and 
    database file within that folder. The database file will contain a
    .json file as a NO-SQL database. This also creates a .refl file in
    this directory that contains information about the location of the
    .refl folder and the database file.
"""

import json
import os

import click
import typer

app = typer.Typer()

@app.command()
def init():
    """Initialize the database."""
    initialize_refl()

def initialize_refl():
    # Get the current directory
    current_dir = os.getcwd()

    # Define the folder and database file names
    refl_folder = ".refl"
    database_file = "refl_database.json"
    refl_file = ".refl"

    # Create the full paths
    refl_folder_path = os.path.join(current_dir, refl_folder)
    database_file_path = os.path.join(refl_folder_path, database_file)
    refl_file_path = os.path.join(current_dir, refl_file)

    # Check if .refl folder and database file already exist
    if os.path.exists(refl_folder_path) and os.path.exists(database_file_path):
        typer.echo("Database and .refl folder already exist.")
        return

    # Create .refl folder if it doesn't exist
    if not os.path.exists(refl_folder_path):
        os.makedirs(refl_folder_path)
        typer.echo(f"Created {refl_folder} folder.")

    # Initialize the database (you can use your own logic here)
    # For example, create an empty JSON file for the database
    if not os.path.exists(database_file_path):
        with open(database_file_path, "w") as db_file:
            json.dump({}, db_file)
        typer.echo(f"Initialized the database: {database_file}")

    # Create .refl file to store information
    refl_info = {
        "refl_folder_path": refl_folder_path,
        "database_file_path": database_file_path,
    }

    with open(refl_file_path, "w") as refl_info_file:
        json.dump(refl_info, refl_info_file)

    typer.echo(f"Created {refl_file} file with path information.")

if __name__ == "__main__":
    app()
