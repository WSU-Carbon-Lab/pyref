import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Final

ENERGY_OPTIONS: Final = ["energy", "Energy", "ENERGY"]
NEXAFS_OPTIONS: Final = ["fit", "result", "nexafs"]


def containsString(text, string_list):
    """
    Simple function to check if any element in strin_list is contained
    in the text
    """
    return any(string in text for string in string_list)


def expLoader(igor_file: Path) -> list[pd.DataFrame]:
    """
    Function manages unpacking a single large csv into into dataframe pairs
    of energy and nexafs spectrum. This also each output as a csv in the
    working directory.

    Parameters
    ----------
    igor_file : Path
        path to the csv file that should contain an even number of columns
        with labels of energy and something resembling the nexafs spectrum.

    Returns
    -------
    list[pd.DataFrame]
        This returns a list of pandas dataframes containing the nexafs
        spectrum for each incidence angle.
    """
    global ENERGY_OPTIONS
    global NEXAFS_OPTIONS

    unpacked_file = pd.read_csv(igor_file)
    assert unpacked_file.columns.size // 2 == unpacked_file.columns.size / 2

    en_cols = [
        col for col in unpacked_file.columns if containsString(col, ENERGY_OPTIONS)
    ]

    nx_cols = [
        col for col in unpacked_file.columns if containsString(col, NEXAFS_OPTIONS)
    ]

    NEXAFS_list = []
    for i, (en, nx) in enumerate(zip(en_cols, nx_cols)):
        singleNEXAFS_df = unpacked_file.loc[:, [en, nx]]
        singleNEXAFS_df.to_csv(Path(igor_file).parent / f"expTypeNEXAFS_{i}.csv")
        NEXAFS_list.append(singleNEXAFS_df)

    return NEXAFS_list


def dftLoader(igor_dataframe: Path) -> pd.DataFrame:
    raise NotImplementedError


def piStarTransition(magic_ang_NEXAFS: pd.DataFrame) -> tuple:
    """
    Uses the scipy find peaks backend to locate the peaks in one of the
    generated NEXAFS dataframes.

    Parameters
    ----------
    magic_ang_NEXAFS : pd.DataFrame
        dataframe for the magic angle nexafs

    Returns
    -------
    tuple
        Tuple containing the peak location for the pi star transition
        allong with its index. If no peaks are found, this returns None
    """
    last_col_ma = magic_ang_NEXAFS.columns[-1]

    signal = magic_ang_NEXAFS[last_col_ma]
    peaks, _ = find_peaks(signal)

    for i, peak in enumerate(
        peaks
    ):  # this filter is needed to throw out low energy things
        if peak < 10:
            peaks.pop(i)

    if len(peaks) > 0:
        piStar_Loc = peaks[0]
        piStar_Amp = signal[piStar_Loc]
        return piStar_Amp, piStar_Loc
    else:
        return None, None


def findDichroism(high_NEXAFS: pd.DataFrame, low_NEXAFS: pd.DataFrame) -> pd.DataFrame:
    """
    Subtracts the low value nexafs from the high value nexafs computing
    the Dichroism.

    Parameters
    ----------
    high_NEXAFS : pd.DataFrame
        high angle nexafs.
    low_NEXAFS : pd.DataFrame
        low angle nexafs.

    Returns
    -------
    pd.DataFrame
        computed Dichroism.

    Raises
    ------
    ValueError
        Ensure that each input data frame is of the same size.
    """
    if len(high_NEXAFS) != len(low_NEXAFS):
        raise ValueError("Dataframes must have the same length.")

    last_col_h = high_NEXAFS.columns[-1]
    first_col_h = high_NEXAFS.columns[0]
    last_col_l = low_NEXAFS.columns[-1]

    dichroism = high_NEXAFS[last_col_h] - low_NEXAFS[last_col_l]
    dichroism_df = pd.DataFrame(
        {"Energy": high_NEXAFS[first_col_h], "Dichroism": dichroism}
    )
    return dichroism_df


def findDichroismScale(
    magic_ang_NEXAFS: pd.DataFrame, dicrhoism: pd.DataFrame
) -> float:
    """
    Determines the number of times that the dicrhoism needs to be added to
    the magic angle NEXAFS untill the first peak is 3x it's original
    amplitude.

    Parameters
    ----------
    magic_ang_NEXAFS : pd.DataFrame
        _description_
    dicrhoism : pd.DataFrame
        _description_

    Returns
    -------
    float
        _description_
    """

    initial_amp, peak_loc = piStarTransition(magic_ang_NEXAFS)
    scale = (initial_amp * 3) / dicrhoism["Dichroism"][peak_loc]

    if scale > 0:
        return scale
    else:
        return -1 * scale


def computeOpticalConstants(
    magic_ang_NEXAFS: pd.DataFrame, dicrhoism: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the optical constants by adding and subtracting the dicrhoism
    from the magic angle nexafs.

    Parameters
    ----------
    magic_ang_NEXAFS : pd.DataFrame
        magic angle nexafs
    dicrhoism : pd.DataFrame
        dichroism

    Returns
    -------
    pd.DataFrame
        Dataframe containing the oriented optical constants along with
        the magic angle nexafs.
    """

    scale = findDichroismScale(magic_ang_NEXAFS, dicrhoism)
    last_col_ma = magic_ang_NEXAFS.columns[-1]

    beta_perp = magic_ang_NEXAFS[last_col_ma] + scale * dicrhoism["Dichroism"]
    beta_para = magic_ang_NEXAFS[last_col_ma] - scale * dicrhoism["Dichroism"]

    magic_ang_NEXAFS["Beta_⟂"] = beta_perp
    magic_ang_NEXAFS["Beta_∥"] = beta_para

    return magic_ang_NEXAFS


if __name__ == "__main__":
    df_list = expLoader(
        Path(
            r"C:\Users\Harlan Heilman\Washington State University (email.wsu.edu)\Carbon Lab Research Group - Documents (1)\Harlan Heilman\DFT-Calculations-of-Organic-Molecules\znpcClusteredDFT_0.csv"
        )
    )
    magic_ang = df_list[int(len(df_list) // 2 + 1)]
    dichroism = findDichroism(df_list[0], df_list[-1])

    oriented_beta = computeOpticalConstants(magic_ang, dichroism)

    oriented_beta.plot(x="expEnergy4_alpha55", y="Beta_∥")
    plt.show()
