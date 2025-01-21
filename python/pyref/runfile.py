"""
# XRR Runfile Generator.

This module is used to generate a runfile for the XRR experiment at the ALS beamline
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml


def unique_filename(path: Path) -> Path:
    """Generate a unique filename."""
    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix
    if "(" in stem and ")" in stem:
        base, num = stem.rsplit("(", 1)
        num = num.rstrip(")")
        if num.isdigit():
            num = int(num) + 1
        else:
            num = 1
    else:
        base = stem
        num = 1

    new_path = path.parent / f"{base}({num}){suffix}"
    return unique_filename(new_path)


def load_config(config: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parse the config.yaml file."""
    with config.open("rb") as f:
        config = yaml.safe_load(f)

    df = pl.DataFrame({str(k): v for k, v in config["energy"].items()}, strict=False)
    return df, config["config"]


def process_stitch(
    theta_i, theta_f, hos, hes, energy, et, points_per_fringe, fringe_size, x
) -> pd.DataFrame:
    """Process a stitch."""
    # Use q space instead of theta space
    q_i = np.sin(np.deg2rad(theta_i)) * 4 * np.pi / (12.4 / energy)
    q_f = np.sin(np.deg2rad(theta_f)) * 4 * np.pi / (12.4 / energy)

    n_points = int(np.ceil((q_f - q_i) / fringe_size) * points_per_fringe)

    q = np.linspace(q_i, q_f, n_points)
    theta = np.rad2deg(np.arcsin(q * (12.4 / energy) / (4 * np.pi))).round(2)
    ccd = 2 * theta.round(2)
    hos = np.full_like(theta, hos).round(2)
    hes = np.full_like(theta, hes).round(0)
    et = np.full_like(theta, et).round(3)
    x = np.full_like(theta, x).round(3)
    energy = np.full_like(theta, energy).round(2)

    return pl.DataFrame(
        {
            "theta": theta,
            "ccd": ccd,
            "hos": hos,
            "hes": hes,
            "et": et,
            "x": x,
            "energy": energy,
        }
    )


def add_overlap(
    current_df: pl.DataFrame, last_df: pl.DataFrame, overlap: int, repeat: int
) -> pd.DataFrame:
    """Add overlap to the current_df."""
    # Add back the overlap points
    overlap_df = last_df[-overlap:]
    overlap_df["hos"] = current_df[0, "hos"]
    overlap_df["hes"] = current_df[0, "hes"]
    overlap_df["et"] = current_df[0, "et"]
    # In the overlap_df repeat the first point repeat times
    overlap_df = pl.concat([overlap_df[0:1]] * repeat)
    return pl.concat([overlap_df, current_df])


def process_energy(df_slice: pd.DataFrame, config: dict, energy: float) -> pd.DataFrame:
    """Generate a chunk for a single energy."""
    theta = df_slice["theta"]
    hos = df_slice["hos"]
    hes = df_slice["hes"]
    et = df_slice["et"]
    n_izero = config["collection"]["izero"]
    x = config["geometry"]["x"]

    # This is parameterized by the number of points in a fringe
    points_per_fringe = config["collection"]["density"]
    fringe_size = 2 * np.pi / config["collection"]["thickness"]

    theta_pairs = [(theta[i], theta[i + 1]) for i in range(len(theta) - 1)]
    energy_df = []
    for i, (t_i, t_f) in enumerate(theta_pairs):
        x += i * 0.1
        stitch_df = process_stitch(
            t_i, t_f, hos[i], hes[i], energy, et[i], points_per_fringe, fringe_size, x
        )
        energy_df.append(stitch_df)

    energy_df = pl.concat(energy_df)
    energy_df = energy_df.with_columns(
        pl.lit(config["geometry"]["z"], dtype=pl.Float64).alias("z"),
    )
    izero = pl.DataFrame(
        {
            "theta": [0.0] * n_izero,
            "ccd": [0.0] * n_izero,
            "hos": [config["izero"]["hos"]] * n_izero,
            "hes": [float(config["izero"]["hes"])] * n_izero,
            "et": [config["izero"]["et"]] * n_izero,
            "x": [config["geometry"]["x"]] * n_izero,
            "energy": [energy] * n_izero,
            "z": [config["izero"]["z"]] * n_izero,
        },
        schema_overrides={
            "theta": pl.Float64,
            "ccd": pl.Float64,
            "hos": pl.Float64,
            "hes": pl.Float64,
            "et": pl.Float64,
            "x": pl.Float64,
            "energy": pl.Float64,
            "z": pl.Float64,
        },
    )
    energy_df = pl.concat([izero, energy_df])
    return energy_df


def generate_runfile(
    df_stitches, config, macro_folder=str | Path
) -> tuple[pl.DataFrame, str]:
    """
    Generate a run file.

    Parameters
    ----------
    macro_folder : str | Path
        Location of the macro folder

    Returns
    -------
    None

    Outputs a run file macro for the ALS beamline 11.0.1.2 at the Advanced Light Source.

    Example
    -------
    >>> generate_runfile("sample1", path_to_save)
    File
    >>> Sample X --> ... Sample Theta --> CCD Theta --> ... Beamline Energy -->
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>>   ⋮                ⋮                  ⋮                   ⋮                 ⋮
    >>> 12.4     --> ... 70            --> 140         --> ... 319          --> 10

    Columns
    -------
    Sample X : float
        Sample X position

    Sample Y : float
        Sample Y position

    Sample Z : float
        Sample Z position

    Sample Theta : float
        Sample Theta position

    CCD Theta : float
        CCD Theta position

    Higher Order Suppressor : float
        Higher Order Suppressor position

    Horizontal Exit Slit Size: float
        Horizontal Exit Slit position

    Beamline Energy : float
        Beamline Energy

    Exposure Time : float
        Exposure Time - This column has no label in the run file
    """
    energies = ", ".join(df_stitches.columns)
    save_path = Path(macro_folder) / f"{config["name"]}[{''.join(energies)}].txt"
    # Generate a new name if the file allready exists

    df = []
    for _, en in enumerate(df_stitches.columns):
        energy_df = process_energy(df_stitches[en][0], config, float(en))
        y = pl.Series("y", [config["geometry"]["y"]] * len(energy_df))
        energy_df = energy_df.hstack([y])
        df.append(energy_df)

    df = pl.concat(df)
    df = df.select(
        pl.col("x"),
        pl.col("y"),
        pl.col("z"),
        pl.col("theta"),
        pl.col("ccd"),
        pl.col("hos"),
        pl.col("hes"),
        pl.col("energy"),
        pl.col("et"),
    )
    df = df.rename(
        {
            "x": "Sample X",
            "y": "Sample Y",
            "z": "Sample Z",
            "theta": "Sample Theta",
            "ccd": "CCD Theta",
            "hos": "Higher Order Suppressor",
            "hes": "Horizontal Exit Slit Size",
            "energy": "Beamline Energy",
            "et": "Exposure",
        }
    )
    if save_path.exists():
        # Check if ther are changes in the file
        save_path = unique_filename(save_path)
    # df = df.sort(
    #     ["Sample Theta", "Higher Order Suppressor", "Horizontal Exit Slit Size"]
    # )
    df = pl.DataFrame(df)
    print(df)
    return df, save_path

    # Construct the runfile


def runfile(config: str | Path, data_path: str | Path = Path.cwd()) -> None:
    """
    Constructes the macro for the XRR experiment allong with saving the data.

    Example
    -------
    >>> generate_runfile("sample1", path_to_save)
    File
    >>> Sample X --> ... Sample Theta --> CCD Theta --> ... Beamline Energy -->
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>>   ⋮                ⋮                  ⋮                   ⋮                 ⋮
    >>> 12.4     --> ... 70            --> 140         --> ... 319          --> 10

    Columns
    -------
    Sample X : float
        Sample X position

    Sample Y : float
        Sample Y position

    Sample Z : float
        Sample Z position

    Sample Theta : float
        Sample Theta position

    CCD Theta : float
        CCD Theta position

    Higher Order Suppressor : float
        Higher Order Suppressor position

    Horizontal Exit Slit Size: float
        Horizontal Exit Slit position

    Beamline Energy : float
        Beamline Energy

    Exposure Time : float
        Exposure Time - This column has no label in the run file
    """
    # Create the save location for the data
    date = datetime.datetime.now()
    beamtime = f"{date.strftime('%Y%b')}/XRR/"

    df_stitches, config_dict = load_config(config)

    save_path = data_path / beamtime / ".macro"
    # Generate the runfile
    df, name = generate_runfile(df_stitches, config_dict, save_path)
    if not name.parent.exists():
        name.parent.mkdir(parents=True)

    name = unique_filename(name)

    df.write_csv(name, separator="\t")
    # remove the last newline, and the "" column name
    runfile_name = name
    with open(runfile_name, "r") as f:
        lines = f.readlines()

    lines[0] = lines[0].replace("\tExposure", "")  # Remove the 'Exposure' header
    lines[-1] = lines[-1].replace("\n", "")  # Remove the last carriage return

    with open(runfile_name, "w") as f:
        f.writelines(lines)
