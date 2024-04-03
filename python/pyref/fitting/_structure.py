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

structure_params = ["thickness", "roughness", "density"]

editor_options = {"nano":["GNU.nano"], "vim":["vim.vim"], "nvim":["neovim.neovim"], "emacs":["gmu.emacs"],"notepad":["notepad"],"sublime":["SublimeHQ.SublimeText.4"], "code":["Microsoft.VisualStudioCode"], "atom":["GitHub.Atom"], "pycharm":["JetBrains.PyCharm.Community"], "spyder":["Spyder.Spyder"]}

install_commands = {
    "nt": {"exe": ["choco", "winget"],
           "command": "install",
           },
    "posix": {"exe": ["sudu apt-get"],
              "command": "install",},
    "darwin": {"exe": ["brew"],
                "command": "install",},
}

def platforms():
    pf = sys.platform
    if pf.startswith("win"):
        return "nt"
    elif pf.startswith("linux"):
        return "posix"
    elif pf.startswith("darwin"):
        return "darwin"
    else:
        return "posix"
    


app = typer.Typer(rich_markup_mode='markdown', pretty_exceptions_show_locals=False)

''' 
Struct command
'''
def check_struct(structure):
    if "layers" not in structure.keys():
        typer.secho("Layers missing from the structure", fg=typer.colors.RED, bold=True)
        typer.secho("Input Chemical formula for each layer in the model\n")
        typer.secho("[Vacuum] | [layerN] | ... | [layer1] | [C] | [SiO2] | [Si]\n".center(typer.get_terminal_size().columns), fg=typer.colors.BRIGHT_MAGENTA, bold=True)
        layer_str = typer.prompt("Enter the name layers in the model", type=str, default = "")
        if layer_str == "":
            layers = ["Vacuum","C","SiO2","Si"]
        else:
            layers = layer_str.split(',')
        yn = typer.confirm(f"Layers: {layers}")
        if yn:
            structure["layers"] = layers
            for layer in layers:
                structure[layer] = {}
                if layer == "Vacuum" or layer == "Si":
                    structure[layer]["density"] = 0
                    structure[layer]["roughness"] = 0 if layer == "Vacuum" else 1.5
                    structure[layer]["thickness"] = 0

            structure_gen_interface(structure)
        else:
            check_struct(structure)

def check_single_layer(structure:dict, layer: str):
    if layer == "Vacuum" or layer == "Si":
            pass
    
    typer.secho(f"Layer: {layer}", fg=typer.colors.BRIGHT_BLUE, bold=True)

    if "thickness" not in structure[layer].keys():
        typer.secho("Thickness missing from the structure", fg=typer.colors.RED, bold=True)
        thickness = typer.prompt("Enter the thickness of the layer", type=float)
        structure[layer]["thickness"] = thickness
        structure_gen_interface(structure) 
    
    else:
        typer.secho(f"Thickness: {structure[layer]['thickness']}", fg=typer.colors.BLACK, bold=True)

    if "roughness" not in structure[layer].keys():
        typer.secho("Roughness missing from the structure", fg=typer.colors.BLACK, bold=True)
        roughness = typer.prompt("Enter the roughness of the layer", type=float, default=structure[layer]["thickness"]/2)
        structure[layer]["roughness"] = roughness
        structure_gen_interface(structure) 
    
    else:
        typer.secho(f"Roughness: {structure[layer]['roughness']}", fg=typer.colors.BLACK, bold=True)

    if "density" not in structure[layer].keys():
        typer.secho("Density missing from the structure", fg=typer.colors.RED, bold=True)
        density = typer.prompt("Enter the density of the layer", type=float, default=0)
        structure[layer]["density"] = density
        structure_gen_interface(structure) 
    
    else:
        typer.secho(f"Density: {structure[layer]['density']}", fg=typer.colors.BLACK, bold=True)

    typer.secho("-"*typer.get_terminal_size().columns, fg=typer.colors.BLACK, bold=True)

def check_layer(structure: dict):
    for layer in structure["layers"]:
        check_single_layer(structure, layer)

def is_executable_on_path(executable_name: str) -> bool:
    return shutil.which(executable_name) is not None

def install_editor(editor: str, exe, install_command):
    if exe == "winget":
        editor = editor_options[editor][0]
    
    typer.secho(f"Installing {editor}...", fg=typer.colors.GREEN, bold=True)
    out = subprocess.run([exe, install_command, editor])
    if out.returncode == 0:
        typer.secho(f"{editor} installed", fg=typer.colors.GREEN, bold=True)
    
def write_json(path, editor, generate = True):
    path = Path(path).absolute()
    typer.secho(f"Locating structure file... \n", fg=typer.colors.GREEN, bold=True)
    if generate == True:
        example = Path("example_struct.json")
        shutil.copy2(example, path)
    
    typer.pause(f"Compleated! press enter to edit...")
    pf = platforms()

    install_exe = install_commands[pf]["exe"]
    install_command = install_commands[pf]["command"]

    if editor == None:
        editor = typer.prompt(f"Editor", default="code")

    if is_executable_on_path(editor):
        subprocess.run([editor, str(path)])

    elif is_executable_on_path(install_exe[0]):
        install_editor(editor, install_exe[0], install_command)
        subprocess.run([editor, str(path)])

    elif len(install_exe) > 1 and is_executable_on_path(install_exe[1]):
        install_editor(editor, install_exe[1], install_command)
        subprocess.run([editor, str(path)])

    else:
        typer.secho(f"Please install {editor} and try again", fg=typer.colors.RED, bold=True)
    
def structure_gen_interface(structure):
    typer.clear()
    typer.secho(art.text2art("pyref - Structure"), fg = typer.colors.YELLOW, bold = True)
    typer.secho("-"*typer.get_terminal_size().columns, fg=typer.colors.BLACK, bold=True)
    typer.secho('\nInterface for creating and editing the slab structure for the fitting analysis\n', fg=typer.colors.YELLOW, bold=True)

    check_struct(structure)
    typer.secho(f"Slab Structure: {" | ".join(structure["layers"])}", fg=typer.colors.BRIGHT_MAGENTA, bold=True)

    typer.secho("-"*typer.get_terminal_size().columns, fg=typer.colors.BLACK, bold=True)
    typer.secho('\nChecking Layer Parameters\n', fg=typer.colors.YELLOW, bold=True)
    check_layer(structure)

@app.command()
def generate(
    save_dir: Annotated[
        str, 
        typer.Option(
            "--save", 
            "-s", 
            help='Save location for a generated model', 
            exists=True, 
            show_default=False,
            rich_help_panel="Options",
            ), 
        None
    ] = None, # type: ignore
    name: Annotated[
        str, 
        typer.Option(
            "--name", 
            "-n", 
            help='Name of the structure file (.json) ',
            exists=False, 
            show_default=False,
            rich_help_panel="Options",
            ),
        None
    ] = None, # type: ignore
    touch: Annotated[
        bool, 
        typer.Option(
            "--touch", 
            "-t", 
            help='Create a structure.json file and open it in an editor', 
            exists=False, 
            show_default=False,
            rich_help_panel="Manual Input",
            ), 
        False
    ] = False, # type: ignore
    editor: Annotated[
        str, 
        typer.Option(
            "--editor", 
            "-e", 
            help='Editor to use for editing the structure file (.json). Used with --touch', 
            exists=False, 
            show_default=False,
            rich_help_panel="Manual Input",
            ), 
        None
    ] = None, # type: ignore
):
    """ 
    **Generate** a slab model for fitting analysis :sparkles:

    Interface for creating a new slab structure.

    --- 
    
    :sparkles: **Each Layer Contains** :sparkles:

    * Thickness

    * Roughness 

    * Density
    """
    
    if save_dir == None:
            save_dir = "./"
    if name == None:
        sample_name = typer.prompt("Enter the name of the sample", type=str)
    else:
        sample_name = name
    file_loc = f"{save_dir}{sample_name}.json"
    if touch == True:
        write_json(file_loc, editor)
        print(f":sparkles: [bold green]STRUCTURE SAVED TO {file_loc}[/bold green] :sparkles:")

    else:
        typer.clear()
        structure = {}
        structure_gen_interface(structure)
        correct_structure = typer.confirm("Is the structure correct?")
        if correct_structure:
            with open(file_loc, 'w') as f:
                json.dump(structure, f, indent=4)

            print(f":sparkles: [bold green]STRUCTURE SAVED TO {file_loc}[/bold green] :sparkles:")

        else:
            typer.secho("Please re-enter the structure", fg=typer.colors.RED, bold=True)
            structure_gen_interface(structure)

@app.command()
def edit(
    load: Annotated[
        str, 
        typer.Option(
            "--load", 
            "-l", 
            help='Load a pre generated structure file (.json) ', 
            exists=True, 
            show_default=False,
            rich_help_panel="Interface Type",
            ), 
        None
    ] = None, # type: ignore
    touch: Annotated[
        bool, 
        typer.Option(
            "--touch", 
            "-t", 
            help='Create a structure.json file and open it in an editor', 
            exists=False, 
            show_default=False,
            rich_help_panel="Manual Editing",
            ), 
        False
    ] = False, # type: ignore
    editor: Annotated[
        str, 
        typer.Option(
            "--editor", 
            "-e", 
            help='Editor to use for editing the structure file (.json). Used with --touch', 
            exists=False, 
            show_default=False,
            rich_help_panel="Manual Editing",
            ), 
        None
    ] = None, # type: ignore
):
    """ 
    **Edit** an existing slab model for fitting analysis :wrench:
    
    * Interface for editing a pre generated slab structure.

    --- 
    
    :sparkles: **Each Layer Contains** :sparkles:

    * Thickness

    * Roughness 

    * Density
    """
    
    if load != None:
        with open(load, 'r') as f:
            structure = json.load(f)
        
        if touch == True:
            write_json(load, editor, generate=False)
            print(f":sparkles: [bold green]STRUCTURE SAVED TO {load}[/bold green] :sparkles:\n")
            typer.secho(f"Validating ...", fg=typer.colors.GREEN, bold=True)
            structure_gen_interface(structure)
        
        else:
            typer.secho(f"Structure loaded from {load}", fg=typer.colors.GREEN, bold=True)
            print(structure)
            structure_gen_interface(structure)
        
        print(f":sparkles: [bold green]STRUCTURE VALIDATED[/bold green] :sparkles:")
    
    else:
        typer.secho("See struct --help", fg=typer.colors.RED, bold=True)

@app.command()
def validate(
    load: Annotated[
        str, 
        typer.Option(
            "--load", 
            "-l", 
            help='Load a pre generated structure file (.json) ', 
            exists=True, 
            show_default=False,
            rich_help_panel="Interface Type",
            ), 
        None
    ] = None, # type: ignore
):
    """ 
    **Validate** a slab model for fitting analysis :heavy_check_mark:

    Interface for validating a pre generated slab structure.

    --- 
    
    :sparkles: **Each Layer Contains** :sparkles:

    * Thickness

    * Roughness 

    * Density
    """
    
    if load != None:
        with open(load, 'r') as f:
            structure = json.load(f)
        
        typer.secho(f"Structure loaded from {load}", fg=typer.colors.GREEN, bold=True)
        structure_gen_interface(structure)
        print(f":sparkles: [bold green]STRUCTURE VALIDATED[/bold green] :sparkles:")
    
    else:
        typer.secho("See struct --help", fg=typer.colors.RED, bold=True)
        


if __name__ == "__main__":
    app()