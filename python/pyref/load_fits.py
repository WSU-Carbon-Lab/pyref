""" 
Im sorry to whoever has to read this code.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from email.mime import image
from multiprocessing import Pool
from pathlib import Path
from shutil import copy2

import art
import click
import numpy as np
import pandas as pd
from astropy.io import fits

from ._config import FLAGS, HEADER_DICT, HEADER_LIST
from .toolkit import FileDialog

DATA_PATH = (
    Path("Washington State University (email.wsu.edu)")
    / "Carbon Lab Research Group - Documents"
    / "Synchrotron Logistics and Data"
    / "ALS - Berkeley"
    / "Data"
    / "BL1101"
)
date = datetime.datetime.now()
BEAMTIME = date.strftime("%Y%b")
DAY = date.strftime("%Y %m %d")


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class FitsReader:
    @staticmethod
    def readHeader(fitsFilePath: Path, headerValues: list[str] = HEADER_LIST) -> dict:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore

        return {
            HEADER_DICT[key]: round(headerData[key], 4)
            if isinstance(headerData[key], (int, float))
            else headerData[key]
            for key in headerValues
            if key in headerData
        }

    @staticmethod
    def readImage(fitsFilePath: Path) -> list:
        with fits.open(fitsFilePath) as hdul:
            imageData = hdul[2].data  # type: ignore

        return imageData

    @staticmethod
    def readFile(
        fitsFilePath: Path, headerValues: list[str] = HEADER_LIST
    ) -> tuple[pd.DataFrame, list]:
        with fits.open(fitsFilePath) as hdul:
            headerData = hdul[0].header  # type: ignore
            imageData = hdul[2].data  # type: ignore

        headerDict = {
            HEADER_DICT[key]: round(headerData[key], 4)
            for key in headerValues
            if key in headerData
        }
        return pd.DataFrame(headerDict, index=[0]), imageData


class MultiReader:
    @staticmethod
    def readHeader(
        dataFilePath: Path,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
    ) -> pd.DataFrame:
        headerDFList = []
        for file in dataFilePath.glob("*.fits"):
            headerDF = pd.DataFrame(
                FitsReader.readHeader(file, headerValues=headerValues), index=[0]
            )
            if fileName:
                headerDF["File Path"] = file.name
            headerDFList.append(headerDF)

        return pd.concat(headerDFList)

    @staticmethod
    def readImage(dataFilePath: Path) -> list:
        imageList = []
        for file in dataFilePath.glob("*.fits"):
            imageList.append(FitsReader.readImage(file))

        return imageList

    @staticmethod
    def readFile(
        dataFilePath: Path | None,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
    ) -> tuple[pd.DataFrame, list]:
        if dataFilePath == None:
            raise ValueError("Restart operation and choose a file path")
        imageList = []
        headerDFList = []
        for i, file in enumerate(dataFilePath.glob("*.fits")):
            headerDF = pd.DataFrame(
                FitsReader.readHeader(file, headerValues=headerValues), index=[i]
            )
            if fileName:
                headerDF["File Path"] = file.name
            headerDFList.append(headerDF)
            imageList.append(FitsReader.readImage(file))

        return pd.concat(headerDFList), imageList

    @staticmethod
    def prepareReflData(
        dataFilePath: Path | None,
        headerValues: list[str] = HEADER_LIST,
        fileName: bool = False,
    ) -> tuple[list[pd.DataFrame], list[list]]:
        if dataFilePath == None:
            raise ValueError("Restart operation and choose a file path")
        pol = dataFilePath.name
        en = dataFilePath.parent.name
        scan_number = dataFilePath.parent.parent.stem
        sample = dataFilePath.parent.parent.parent.stem
        sample_path = dataFilePath.parent.parent.parent

        file_name = f"{sample}_{en}_{pol} ({scan_number})"
        if (sample_path / f"{file_name}.npz").exists():
            loaded_images = np.load(sample_path / f"{file_name}.npz")
            images = [loaded_images[key] for key in loaded_images.keys()]
            meta_data = pd.read_parquet(sample_path / f"{file_name}.parquet")
        else:
            meta_data, images = MultiReader.readFile(
                dataFilePath, headerValues=headerValues, fileName=fileName
            )
            meta_data.to_parquet(sample_path / f"{file_name}.parquet")
            np.savez(sample_path / f"{file_name}.npz", *images)

        headerStitchedDFList = []
        imageStitchedList = []
        j = 0
        for i, _ in enumerate(images):
            if (
                i == 0
                or meta_data["Theta"].iloc[i] >= meta_data["Theta"].iloc[i - 1]
                and i < len(images) - 1
            ):
                continue
            else:
                if i == len(images) - 1:
                    i += 1
                stitch_df = meta_data.iloc[j:i].copy()
                headerStitchedDFList.append(stitch_df)
                imageStitchedList.append(images[j:i])
                j = i

        return headerStitchedDFList, imageStitchedList


class FitsSorter:
    @staticmethod
    def sortDirectory(dataPath: Path, sortedPath: Path, sortedStructure: str) -> None:
        flags = FitsSorter._getFlags(
            sortedPath=sortedPath, sortedStructure=sortedStructure
        )
        FitsSorter._sortByFlags(dataPath, sortedPath, flags)

    @staticmethod
    def _getFlags(sortedPath: Path, sortedStructure: str) -> list:
        global FLAGS

        structureChildren = sortedStructure.split("/")

        directories = []
        flags = []
        for child in structureChildren:
            if child.startswith("-"):
                flags.append(FLAGS[child])
            else:
                directories.append(child)
            directoryPath = sortedPath / "/".join(directories)
            directoryPath.mkdir(exist_ok=True)
        return flags

    @staticmethod
    def _sortByFlags(dataPath: Path, sortedPath: Path, flags: list):
        headerDF = MultiReader.readHeader(dataPath, headerValues=flags, fileName=True)
        headerDF.reindex(columns=flags)
        with ThreadPoolExecutor() as executor:
            list(
                executor.map(
                    lambda row: copyFile(row[1], sortedPath=sortedPath, flags=flags),
                    headerDF.iterrows(),
                )
            )


def copyFile(row, sortedPath: Path, flags: list) -> None:
    sourceFile: Path = row[-1]  # type: ignore
    targetDirectory = sortedPath

    for flag in flags:
        if flag == FLAGS["-n"]:
            append = _getSampleName(row[flag])
        else:
            append = str(round(row[HEADER_DICT[flag]], 1))

        targetDirectory = targetDirectory / append
        targetDirectory.mkdir(exist_ok=True)

    copy2(sourceFile, targetDirectory)


def _getSampleName(string: str) -> str:
    """General File Structure "Sample-ScanID.fits" this parses sample"""
    fileName = Path(string).name.split(".")
    return fileName[0].split("-")[0]


class DatabaseInterface:
    def __init__(self, experiment_directory: Path) -> None:
        self.experiment_directory = experiment_directory
        self.processed_directory = experiment_directory / "Processed"
        self.day_information: None | pd.DataFrame | dict = None

        if not self.processed_directory.exists():
            self.processed_directory.mkdir()
            print(f"{bcolors.OKGREEN}Created Processed Directory{bcolors.ENDC}")

        os.system("cls")
        art.tprint("XRR Sorter")

        self._beamtime = self.experiment_directory.parent.name
        print(f"{bcolors.HEADER}Beamtime: {self._beamtime}{bcolors.ENDC}")
        print("=" * os.get_terminal_size().columns)
        print("/" * os.get_terminal_size().columns)
        print("=" * os.get_terminal_size().columns)

        self.main_menue()
        self.select_day_to_sort()

        os.system("cls")
        art.tprint("XRR Sorter")

        self._day = self.data_path.name
        self._beamtime = self.experiment_directory.parent.name
        print(f"{bcolors.HEADER}{'Beamtime:':<12} {self._beamtime}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{'Day:':<12} {self._day}{bcolors.ENDC}")
        print("=" * os.get_terminal_size().columns)
        print("/" * os.get_terminal_size().columns)
        print("=" * os.get_terminal_size().columns)

        self.scan_menue()

        self.sort_scan()

        restart = input("reastart?")
        if restart in ["y", "Y", "yes", "Yes", "YES"]:
            self.__init__(self.experiment_directory)

    def main_menue(self):
        # TODO: Move to seperate object that can be initialized
        print(f" {'Day:':<12}{'Experiment Name:':<30}{'Number of scans:':<10}")
        for i, directory in enumerate(self.experiment_directory.iterdir()):
            if directory.is_file():
                continue
            n_scans = N_scans(directory)
            if directory.name == "Processed" or directory.name == ".macro":
                print(f"{bcolors.WARNING} {i:<12}{directory.name:<30}{bcolors.ENDC}")
            elif n_scans == 0:
                print(
                    f"{bcolors.FAIL} {i:<12}{directory.name:<30}{n_scans:<10}{bcolors.ENDC}"
                )
            else:
                print(
                    f"{bcolors.OKGREEN} {i:<12}{directory.name:<30}{n_scans:<10}{bcolors.ENDC}"
                )
        print("-" * os.get_terminal_size().columns)

    def select_day_to_sort(self):
        __selection = input(f"Select a directory to sort or enter to restart:")
        if __selection == "":
            self.__init__(self.experiment_directory)

        elif int(__selection) > len(list(self.experiment_directory.iterdir())):
            print("Invalid Selection")
            self.select_day_to_sort()

        _selection = int(__selection)
        selection = list(self.experiment_directory.iterdir())[_selection]
        print(f"Selected: {selection.name}\n")
        self.data_path = selection
        if self.data_path.name == "Processed":
            print("Cannot sort processed data")
            self.__init__(self.experiment_directory)

        elif self.data_path.name == ".macro":
            print("Cannot sort macro files")
            self.__init__(self.experiment_directory)

        elif N_scans(self.data_path) == 0:
            print("No scans in directory")
            self.__init__(self.experiment_directory)

    def scan_menue(self, names: bool = True):
        # TODO: Move to seperate object that can be initialized
        print(
            f" {' ':<4}{'Scan Number:':<20}{'Sample Name':<15}{'Elapsed Time':<20}{'Pol': <12}{'Energies:':<10}"
        )
        if isinstance(self.day_information, type(None)):
            self.day_information = pd.DataFrame(
                columns=[
                    "Scan",
                    "Sample Name",
                    "Pol",
                    "Energies",
                    "Elapsed Time",
                    "Meta Data",
                ]
            )
            self.scans = list(self.data_path.iterdir())  # type: ignore
            df_list = []
            for i, scan in enumerate(self.scans):
                scan_name = scan.name
                if scan_name.startswith("CCD Scan"):
                    series = pd.Series()
                    series["Scan"] = i
                    series["Scan Number"] = scan_name
                    header = HEADER_LIST.append("DATE")
                    meta_data, image_data = MultiReader.readFile(
                        scan / "CCD", fileName=True
                    )
                    series["Meta Data"] = meta_data
                    series["Image Data"] = image_data
                    series["Pol"] = series["Meta Data"]["POL"].iat[0]
                    energies = meta_data["Energy"].round(1).value_counts()  # type: ignore
                    n_fits = energies.to_numpy()

                    series["Energies"] = ", ".join(
                        [f"{en}, ({n_fits[i]})" for i, en in enumerate(energies.index)]
                    )
                    scan_num = scan.name.split(" ")[-1]
                    series["Sample Name"] = (
                        series["Meta Data"]["File Path"]
                        .apply(lambda x: str(x).split(scan_num[0])[0])
                        .iloc[0]
                    )
                    series["Elapsed Time"] = str(
                        series["Meta Data"]["Date"].astype("datetime64[ns]").max()
                        - series["Meta Data"]["Date"].astype("datetime64[ns]").min()
                    ).split(" ")[-1]
                    self.show_scans(series, n_fits)
                    df_list.append(series)
            self.day_information = pd.concat(df_list, axis=1).transpose()
            self.day_information.set_index("Scan", inplace=True)
        else:
            for i, series in self.day_information.iterrows():  # type: ignore
                energies = series["Meta Data"]["Energy"].round(1).value_counts()  # type: ignore
                n_fits = energies.to_numpy()
                self.show_scans(series, n_fits, index=i)

        print("-" * os.get_terminal_size().columns)
        if names:
            self.ensure_names()

    def ensure_names(self):
        correct = input("Are all sample names correct? [y, n]")
        if correct in ["y", "Y", "yes", "Yes", "YES"]:
            os.system("cls")
            art.tprint("XRR Sorter")
            print(f"{bcolors.HEADER}{'Beamtime:':<12} {self._beamtime}{bcolors.ENDC}")
            print(f"{bcolors.HEADER}{'Day:':<12} {self._day}{bcolors.ENDC}")
            print("=" * os.get_terminal_size().columns)
            print("/" * os.get_terminal_size().columns)
            print("=" * os.get_terminal_size().columns)
            self.scan_menue(names=False)

        elif correct in ["n", "N", "no", "No", "NO"]:
            file_num = input("Enter the scan number to change: ")
            old_sample_name = self.day_information["Sample Name"].iloc[int(file_num)]  # type: ignore
            print(f"Current Sample Name: {old_sample_name}")
            print(f"The following scans share this name:")
            similar_scans = self.day_information[  # type: ignore
                self.day_information["Sample Name"] == old_sample_name  # type: ignore
            ]
            print(similar_scans["Scan Number"].to_string())
            apply_to_all = input("Apply to all? [y, n]")
            new_sample_name = input("Enter the new sample name: ")
            if apply_to_all in ["y", "Y", "yes", "Yes", "YES"]:
                for i in similar_scans.index:
                    scan = self.scans[i]
                    scan_num = scan.name.split(" ")[-1]
                    self.rename_fits(scan, new_sample_name, scan_num)
                    self.day_information["Sample Name"].iloc[i] = new_sample_name  # type: ignore
            elif apply_to_all in ["n", "N", "no", "No", "NO"]:
                scan = self.scans[int(file_num)]
                scan_num = scan.name.split(" ")[-1]
                self.rename_fits(scan, new_sample_name, scan_num)
                self.day_information["Sample Name"].iloc[int(file_num)] = new_sample_name  # type: ignore
            elif apply_to_all == "":
                self.ensure_names()
            else:
                print("Invalid input")
                self.ensure_names()

            os.system("cls")
            art.tprint("XRR Sorter")
            print(f"{bcolors.HEADER}{'Beamtime:':<12} {self._beamtime}{bcolors.ENDC}")
            print(f"{bcolors.HEADER}{'Day:':<12} {self._day}{bcolors.ENDC}")
            print("=" * os.get_terminal_size().columns)
            print("/" * os.get_terminal_size().columns)
            print("=" * os.get_terminal_size().columns)
            self.scan_menue()
        elif correct == "":
            self.__init__(self.experiment_directory)
        else:
            print("Invalid input")
            self.ensure_names()

    def rename_fits(self, scan: Path, sample_name: str, scan_num: str):
        ccd = scan / "CCD"
        for file in ccd.iterdir():
            if file.name.endswith(".fits"):
                image_number = file.name.split("-")[-1].split(".")[0]
                new_name = f"{sample_name}{scan_num}-{image_number}.fits"
                file.rename(file.parent / new_name)

    def show_scans(self, series, n_fits, index=None):
        if index != None:
            if np.any(n_fits < 12):
                print(
                    f"{bcolors.FAIL} {index:<4}{series['Scan Number']:<20}{series['Sample Name']:<15}{series['Elapsed Time']:<20}{series['Pol']:<12}{series['Energies']:<10}{bcolors.ENDC}"  # type: ignore
                )
            else:
                print(
                    f"{bcolors.OKGREEN} {index:<4}{series['Scan Number']:<20}{series['Sample Name']:<15}{series['Elapsed Time']:<20}{series['Pol']:<12}{series['Energies']:<10}{bcolors.ENDC}"  # type: ignore
                )
        else:
            if np.any(n_fits < 12):
                print(
                    f"{bcolors.FAIL} {series['Scan']:<4}{series['Scan Number']:<20}{series['Sample Name']:<15}{series['Elapsed Time']:<20}{series['Pol']:<12}{series['Energies']:<10}{bcolors.ENDC}"  # type: ignore
                )
            else:
                print(
                    f"{bcolors.OKGREEN} {series['Scan']:<4}{series['Scan Number']:<20}{series['Sample Name']:<15}{series['Elapsed Time']:<20}{series['Pol']:<12}{series['Energies']:<10}{bcolors.ENDC}"  # type: ignore
                )

    def _sort_scan(self, scan_info: pd.Series):
        target_dir = scan_info["Target"]
        ccd_path = self.data_path / scan_info["Scan Number"] / "CCD"
        energies = scan_info["Energies"].split(",")[::2]
        scan_name = scan_info["Scan Number"]
        pol = str(scan_info["Pol"])
        with click.progressbar(
            enumerate(energies),
            label=f"Sorting Scan {bcolors.WARNING}{scan_name}{bcolors.ENDC}:",
        ) as bar:
            for i, energy in bar:
                mask = (
                    scan_info["Meta Data"]["Energy"].round(1) == float(energy)
                ).to_numpy(dtype=bool)
                df = scan_info["Meta Data"][mask]
                images = np.asarray(scan_info["Image Data"])[mask]
                en = energy.replace(" ", "")
                for j, file in df.iterrows():
                    file_path = ccd_path / file["File Path"]
                    target_path = target_dir / en / pol
                    if not target_path.exists():
                        target_path.mkdir(parents=True)
                    copy2(file_path, target_path)
                saved_df = df.drop(columns=["Date", "File Path"], axis=1)
                assert len(images) == len(saved_df)
                saved_df.to_parquet(
                    target_dir.parent
                    / f"{target_dir.parent.stem}_{en}_{pol} ({scan_name}).parquet"
                )
                np.savez(
                    target_dir.parent
                    / f"{target_dir.parent.stem}_{en}_{pol} ({scan_name}).npz",
                    *images,
                )

    def sort_scan(self):
        # TODO: Set this up so the folders are initialized before the sorting begins
        print("Sorting Scans...")
        rows = []
        for i, row in self.day_information.iterrows():
            sample_directory = self.processed_directory / row["Sample Name"]
            if not sample_directory.exists():
                sample_directory.mkdir()

            processed_scan_dir = sample_directory / row["Scan Number"]
            if not processed_scan_dir.exists():
                processed_scan_dir.mkdir()

            row["Target"] = processed_scan_dir
            rows.append(row)
        with Pool() as pool:
            pool.map(self._sort_scan, rows)


def N_scans(data_dir: Path):
    n_scans = len(
        [dir for dir in data_dir.iterdir() if dir.name.startswith("CCD Scan")]
    )
    return n_scans


@click.command()
@click.option("--beam_time", "-bt", default=BEAMTIME, help="YearMonth of the beamtime")
def cli(
    beam_time,
):
    os.system("cls")
    experiment_directory = Path.home() / DATA_PATH / beam_time / "XRR"
    interface = DatabaseInterface(experiment_directory)


if __name__ == "__main__":
    cli()
