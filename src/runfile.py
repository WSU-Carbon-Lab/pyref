"""
XRR Run File Creation Script
============================
This script is used to create a run file for the XRR macro. 
It will create a folder for the current experiment 
and create a run macro for a single sample based on it's geometry.
"""

import datetime
from typing import Literal
from pathlib import Path
import numpy as np
import warnings as warn
import pandas as pd
import os

DATA_PATH = (
    Path("Washington State University (email.wsu.edu)")
    / "Carbon Lab Research Group - Documents"
    / "Synchrotron Logistics and Data"
    / "ALS - Berkeley"
    / "Data"
    / "BL1101"
)

C = 299792458  # m/s
H_J = 6.6267015e-34  # Joule s
E = 1.602176634e-19  # coulombs
H = H_J / E  # eV s
m_to_a = 10 ** (10)

ANGLE_TRANSITIONS = {
    "short_s": [0, 5, 10, 15],
    "long_s": [0, 5, 10, 12, 20, 25, 40, 50, 70],
    "short_p": [0, 5, 10, 15],
    "long_p": [0, 5, 10, 12, 20, 25, 30, 70],
}
HES = {
    "low": [150],
    "high": [1500],
}

MOTORS = [
    "Sample X",
    "Sample Y",
    "Sample Z",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "Horizontal Exit Slit Size",
    "Beamline Energy",
    "Exposure",
]

standard_params = {
    "EnergyOffset": 0,
    "XOffset": 0,
    "ZFlipPosition": -1.998,
    "ReverseHolder": 0,
    "ZDirectBeam": -2,
    "ThetaOffset": 0,
    "LowAngleDensity": 12,
    "HighAngleDensity": 12,
    "AngleCrossover": 10,
    "OverlapPoints": 3,
    "CheckUncertainty": 4,
    "HOSBuffer": 1,
    "I0Points": 5,
}


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


# Create a path that will create a folder for the current experiment
def make_macro_folder(data_path=DATA_PATH):
    date = datetime.datetime.now()
    beamtime = f"{date.strftime('%Y%b')}/XRR/"
    date = date.strftime("%Y %m %d")

    data_path = Path.home() / data_path
    save_path = data_path / beamtime / ".macro" / date

    if not save_path.exists():
        save_path.mkdir(parents=True)
        print(f"Created {save_path}")

    return save_path


# User Input for Sample Geometry and Setup
def experiment_setup(
    sample_name: str = "Sample1",
    energy_list: list = [250],
    x_position: float = 0,
    y_position: float = 0,
    z_position: float = 0,
    sample_thickness: float = 250,
    parameter_setup: dict = standard_params,
):
    sample1 = pd.DataFrame({sample_name: energy_list})
    motor_position_dict = {
        "XPosition": x_position,
        "YPosition": y_position,
        "ZPosition": z_position,
        "SampleThickness": sample_thickness,
    } | parameter_setup
    motor_position = pd.DataFrame(motor_position_dict, index=[0])
    return sample1, motor_position


# User Input for HOS, HES, and Exposure Time
def energy_setup(
    angle_transitions: Literal["short_s", "long_s", "short_p", "long_p"] | list,
    hos: list,
    et: list,
    hes: Literal["low", "high"] | list | int = "low",
):
    if isinstance(angle_transitions, str):
        angle_transitions = ANGLE_TRANSITIONS[angle_transitions]

    N_transitions = len(angle_transitions) - 1

    if isinstance(hes, str):
        hes = HES[hes] * N_transitions

    if isinstance(hes, int):
        hes = [hes] * N_transitions

    if len([h for h in hos if h < 6.5]) > 0:
        warn.warn("HOS values < 6.5 have no impact on the experiment", UserWarning)

    assert (
        len(hos) == N_transitions
    ), "HOS  must be the same length as the number of angle transitions - 1"
    assert (
        len(et) == N_transitions
    ), "ET must be the same length as the number of angle transitions - 1"

    df = pd.DataFrame([angle_transitions, hos, hes, et]).T
    df.columns = ["Angle", "HOS", "HES", "EXPOSURE"]
    return df


def single_energy_run(sample_positions, stitching_params, energy):
    Wavelength = C * H * m_to_a / energy
    AngleNumber = stitching_params["Angle"].nunique()
    XPosition = sample_positions["XPosition"].iloc[0]
    YPosition = sample_positions["YPosition"].iloc[0]
    ZPosition = sample_positions["ZPosition"].iloc[0]
    Z180Position = sample_positions["ZFlipPosition"].iloc[0]
    Zdelta = ZPosition - Z180Position
    SampleThickness = sample_positions["SampleThickness"]
    LowAngleDensity = sample_positions["LowAngleDensity"]
    OverlapPoints = int(sample_positions["OverlapPoints"].iloc[0])

    for i in range(AngleNumber - 1):
        if i == 0:  # starts the list
            ##Calculate the start and stop location for Q
            AngleStart = stitching_params["Angle"].iloc[
                i
            ]  # All of the relevant values are in terms of angles, but Q is calculated as a check
            AngleStop = stitching_params["Angle"].iloc[i + 1]
            QStart = 4 * np.pi * np.sin(AngleStart * np.pi / 180) / Wavelength
            QStop = 4 * np.pi * np.sin(AngleStop * np.pi / 180) / Wavelength
            AngleDensity = LowAngleDensity

            # Setup dq in terms of an approximate fringe size (L = 2*PI/Thickness)
            # Break it up based on the desired point density per fringe
            dq = 2 * np.pi / (SampleThickness * AngleDensity).iloc[0]
            QPoints = int((QStop - QStart) // dq)
            # Number of points to run is going to depend on fringe size
            QList = np.linspace(
                QStart, QStop, QPoints
            ).tolist()  # Initialize the QList based on initial configuration
            SampleTheta = np.linspace(
                AngleStart, AngleStop, QPoints
            )  ##Begin generating list of 'Sample Theta' locations to take data
            CCDTheta = SampleTheta * 2  # Make corresponding CCDTheta positions
            # Check what side the sample is on. If on the bottom, sample theta starts @ -180
            # if MotorPositions['ReverseHolder'] == 1:
            #    SampleTheta=SampleTheta-180 # for samples on the backside of the holder, need to check and see if this is correct

            SampleX = [XPosition] * len(QList)
            BeamLineEnergy = [energy] * len(QList)
            SampleY = (
                Zdelta / 2 * np.sin(SampleTheta * np.pi / 180) - YPosition - Zdelta / 2
            )
            SampleZ = ZPosition + Zdelta / 2 * np.cos(SampleTheta * np.pi / 180) - 1

            # Convert numpy arrays into lists for Pandas generation
            SampleTheta = SampleTheta.tolist()
            SampleY = SampleY.tolist()
            SampleZ = SampleZ.tolist()
            CCDTheta = CCDTheta.tolist()

            # Generate HOS / HES / Exposure lists for updating flux
            HOSList = [stitching_params["HOS"].iloc[i]] * len(QList)
            HESList = [stitching_params["HES"].iloc[i]] * len(QList)
            ExposureList = [stitching_params["EXPOSURE"].iloc[i]] * len(QList)

            # Adding points to assess the error in beam intensity given new HOS / HES / Exposure conditions
            for d in range(int(sample_positions["I0Points"].iloc[0])):
                QList.insert(0, 0)
                # ThetaInsert = 0 if MotorPositions['ReverseHolder']==0 else -180
                SampleTheta.insert(0, 0)
                CCDTheta.insert(0, 0)
                SampleX.insert(0, SampleX[d])
                SampleY.insert(0, YPosition)
                SampleZ.insert(0, sample_positions["ZDirectBeam"].iloc[0])
                HOSList.insert(0, HOSList[d])
                HESList.insert(0, HESList[d])
                ExposureList.insert(0, ExposureList[d])
                BeamLineEnergy.insert(0, BeamLineEnergy[d])

        else:  # for all of the ranges after the first set of samples
            ##Section is identical to the above
            AngleStart = stitching_params["Angle"].iloc[i]
            AngleStop = stitching_params["Angle"].iloc[i + 1]
            QStart = 4 * np.pi * np.sin(AngleStart * np.pi / 180) / Wavelength
            QStop = 4 * np.pi * np.sin(AngleStop * np.pi / 180) / Wavelength

            AngleDensity = LowAngleDensity

            dq = 2 * np.pi / (SampleThickness * AngleDensity).iloc[0]
            QPoints = int((QStop - QStart) // dq)
            QListAddition = np.linspace(QStart, QStop, QPoints).tolist()
            SampleThetaAddition = np.linspace(AngleStart, AngleStop, QPoints).tolist()
            ##Calculate the points that are used to stitch datasets
            # p+2 selects the appropriate number of points to repeat without doubling at the start of the angle range.
            # Compensate the number of points by reducing OverlapPoints down by 1 (Nominally at 4)
            for p in range(OverlapPoints):
                QListAddition.insert(0, QList[-1 * (p + 2)])  # Add to Qlist
                SampleThetaAddition.insert(
                    0, SampleTheta[-1 * (p + 2)]
                )  # Add to Sample Theta List ###QUICK CHANGE! REMOVE SAMPLE OFFSET TO ADDITION
            SampleThetaAdditionArray = np.asarray(
                SampleThetaAddition
            )  # Convert back to numpy array

            CCDThetaAddition = (
                SampleThetaAdditionArray * 2
            )  # Calculate the CCD theta POsitions
            CCDThetaAddition = CCDThetaAddition.tolist()  # Convert to list
            SampleThetaAddition = SampleThetaAdditionArray.tolist()
            # Check what side the sample is on. If on the bottom, sample theta starts @ -180
            # if MotorPositions['ReverseHolder']==1:
            #    SampleThetaAdditionArray=SampleThetaAdditionArray-180

            SampleXAddition = [XPosition] * len(QListAddition)
            BeamLineEnergyAddition = [energy] * len(QListAddition)
            SampleYAddition = (
                YPosition
                + Zdelta / 2
                + Zdelta / 2 * np.sin(SampleThetaAdditionArray * np.pi / 180)
            )
            SampleZAddition = ZPosition + Zdelta / 2 * (
                np.cos(SampleThetaAdditionArray * np.pi / 180) - 1
            )
            SampleYAddition = SampleYAddition.tolist()
            SampleZAddition = SampleZAddition.tolist()

            # Generate HOS / HES / Exposure lists for updating flux
            HOSListAddition = [stitching_params["HOS"].iloc[i]] * len(QListAddition)
            HESListAddition = [stitching_params["HES"].iloc[i]] * len(QListAddition)
            ExposureListAddition = [stitching_params["EXPOSURE"].iloc[i]] * len(
                QListAddition
            )

            # Check to see if any of the variable motors have moved to add buffer points
            if (
                stitching_params["HOS"].iloc[i] != stitching_params["HOS"].iloc[i - 1]
                or stitching_params["HES"].iloc[i]
                != stitching_params["HES"].iloc[i - 1]
                or stitching_params["EXPOSURE"].iloc[i]
                != stitching_params["EXPOSURE"].iloc[i - 1]
            ):
                # If a change is made, buffer the change with points to judge new counting statistics error and a few points to buffer the motor movements.
                # Motor movements buffer is to make sure motors have fully moved before continuing data collection / may require post process changes
                # Adding points to assess the error in beam intensity given new HOS / HES / Exposure conditions
                for d in range(int(sample_positions["CheckUncertainty"].iloc[0])):
                    QListAddition.insert(0, QListAddition[d])
                    SampleThetaAddition.insert(0, SampleThetaAddition[d])
                    CCDThetaAddition.insert(0, CCDThetaAddition[d])
                    SampleXAddition.insert(0, SampleXAddition[d])
                    SampleYAddition.insert(0, SampleYAddition[d])
                    SampleZAddition.insert(0, SampleZAddition[d])
                    HOSListAddition.insert(0, HOSListAddition[d])
                    HESListAddition.insert(0, HESListAddition[d])
                    ExposureListAddition.insert(0, ExposureListAddition[d])
                    BeamLineEnergyAddition.insert(0, BeamLineEnergyAddition[d])

                # Adding dummy points to beginning of to account for HOS movement
                for d in range(int(sample_positions["HOSBuffer"].iloc[0])):
                    QListAddition.insert(0, QListAddition[d])
                    SampleThetaAddition.insert(0, SampleThetaAddition[d])
                    CCDThetaAddition.insert(0, CCDThetaAddition[d])
                    SampleXAddition.insert(0, SampleXAddition[d])
                    SampleYAddition.insert(0, SampleYAddition[d])
                    SampleZAddition.insert(0, SampleZAddition[d])
                    HOSListAddition.insert(0, HOSListAddition[d])
                    HESListAddition.insert(0, HESListAddition[d])
                    ExposureListAddition.insert(0, ExposureListAddition[d])
                    BeamLineEnergyAddition.insert(0, BeamLineEnergyAddition[d])

            QList.extend(QListAddition)
            HOSList.extend(HOSListAddition)
            HESList.extend(HESListAddition)
            ExposureList.extend(ExposureListAddition)
            SampleTheta.extend(SampleThetaAddition)
            CCDTheta.extend(CCDThetaAddition)
            SampleX.extend(SampleXAddition)
            SampleY.extend(SampleYAddition)
            SampleZ.extend(SampleZAddition)
            BeamLineEnergy.extend(BeamLineEnergyAddition)

        # Check what side the sample is on. If on the bottom, sample theta starts @ -180
    if sample_positions["ReverseHolder"].iloc[0] == 1:
        SampleTheta = [
            theta - 180 for theta in SampleTheta
        ]  # for samples on the backside of the holder, need to check and see if this is correct

    df = pd.DataFrame(columns=MOTORS)
    df["Sample X"] = SampleX  # type: ignore
    df["Sample Y"] = SampleY  # type: ignore
    df["Sample Z"] = SampleZ  # type: ignore
    df["Sample Theta"] = SampleTheta  # type: ignore
    df["Higher Order Suppressor"] = HOSList  # type: ignore
    df["Horizontal Exit Slit Size"] = HESList  # type: ignore
    df["Beamline Energy"] = BeamLineEnergy  # type: ignore
    df["Exposure"] = ExposureList  # type: ignore
    df["CCD Theta"] = CCDTheta  # type: ignore
    df["Q"] = QList  # type: ignore
    return df


def run_file(sample_positions, stitching_params, energy_list):
    total_run = []
    for i, en in enumerate(energy_list):
        samp_en = np.round(en + sample_positions["EnergyOffset"].iloc[0], 2)
        df = single_energy_run(sample_positions, stitching_params[i], samp_en)
        total_run.append(df)
    total_run = pd.concat(total_run)
    total_run = total_run.reset_index(drop=True)
    return total_run


def to_macro(run_file: pd.DataFrame, sample_name, macro_path):
    run_file.drop(columns=["Q"], inplace=True)
    run_file_name = uniquify(f"{macro_path}/{sample_name}.txt")
    run_file.to_csv(run_file_name, index=False, sep="\t")

    ###Cleanup the output -- Definitly better ways to do this....
    with open(run_file_name, "r") as f:  # Loads the file into memory
        lines = f.readlines()

    lines[0] = lines[0].replace("\tExposure", "")  # Remove the 'Exposure' header
    lines[-1] = lines[-1].replace("\n", "")  # Remove the last carriage return

    with open(run_file_name, "w") as f:  # Writes it back in
        f.writelines(lines)

    del lines  # Remove it from memory (it can be large)
