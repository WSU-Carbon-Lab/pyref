from typing import Literal
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from xrr._config import REFL_COLUMN_NAMES
from xrr.image_manager import ImageProcs
from xrr.toolkit import XrayDomainTransform
from xrr.refl_reuse import Reuse


class ReflProcs:
    @staticmethod
    def main(
        obj,
        mask,
        metaData: pd.DataFrame,
        source: Literal[
            "fits",
            "csv",
        ] = "fits",
        *args,
        **kwargs
    ):
        if source == "fits":
            obj.filtered, obj.masked, beamSpots, darkSpots = ReflProcs.getBeamSpots(
                obj.images, mask=mask
            )
            directBeam, backgroundNoise = ReflProcs.getSubImages(
                obj.masked, beamSpots, darkSpots
            )
            pureReflDF = ReflProcs.getDf(metaData, directBeam, backgroundNoise)
            obj.refl = ReflProcs.scaleSeries(pureReflDF, *args, **kwargs)

        elif source == "csv":
            Reuse.openForReuse(obj)

    @staticmethod
    def getBeamSpots(
        imageList: list, mask: np.ndarray | None = None
    ) -> tuple[list, list, list, list]:
        with ThreadPoolExecutor() as executor:
            trimmedImages = list(
                executor.map(lambda image: ImageProcs.removeEdge(image), imageList)
            )
            if mask != None:
                maskedImages = list(
                    executor.map(
                        lambda image: ImageProcs.applyMask(image, mask), trimmedImages
                    )
                )
                filteredImages = list(
                    executor.map(
                        lambda image: ImageProcs.medianFilter(image), maskedImages
                    )
                )
            else:
                maskedImages = trimmedImages
                filteredImages = list(
                    executor.map(
                        lambda image: ImageProcs.medianFilter(image), trimmedImages
                    )
                )
            beamSpots = list(
                executor.map(
                    lambda image: ImageProcs.findMaximum(image), filteredImages
                )
            )
            darkSpots = list(
                executor.map(
                    lambda spot: ImageProcs.oppositePoint(spot, trimmedImages[0].shape),
                    beamSpots,
                )
            )
        return filteredImages, maskedImages, beamSpots, darkSpots

    @staticmethod
    def getSubImages(
        maskedImages: list[np.ndarray],
        beamSpots: list[tuple],
        darkSpots: list[tuple],
        height: int = 20,
        width: int = 20,
    ):
        with ThreadPoolExecutor() as executor:
            directBeam = list(
                executor.map(
                    lambda args: ImageProcs.roiReduction(
                        args[0], args[1], height=height, width=width
                    ),
                    zip(maskedImages, beamSpots),
                )
            )
            backgroundNoise = list(
                executor.map(
                    lambda args: ImageProcs.roiReduction(
                        args[0], args[1], height=height, width=width
                    ),
                    zip(maskedImages, darkSpots),
                )
            )
        return directBeam, backgroundNoise

    @staticmethod
    def getRefl(reflBeamSpots: list, darkBeamSpots: list) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            refl_intensity = list(
                executor.map(lambda image: ImageProcs.sumImage(image), reflBeamSpots)
            )
            dark_intensity = list(
                executor.map(lambda image: ImageProcs.sumImage(image), darkBeamSpots)
            )
        df = pd.DataFrame(
            {
                REFL_COLUMN_NAMES["Beam Spot"]: refl_intensity,
                REFL_COLUMN_NAMES["Dark Spot"]: dark_intensity,
            }
        )
        return df

    @staticmethod
    def getDf(metaData: pd.DataFrame, brightSpots, darkSpots) -> pd.DataFrame:
        reflData = ReflProcs.getRefl(brightSpots, darkSpots)
        reflData.reset_index(drop=True, inplace=True)
        reflData[REFL_COLUMN_NAMES["R"]] = (
            reflData[REFL_COLUMN_NAMES["Beam Spot"]]
            - reflData[REFL_COLUMN_NAMES["Dark Spot"]]
        ) / (
            metaData[REFL_COLUMN_NAMES["Beam Current"]]
            * metaData[REFL_COLUMN_NAMES["Higher Order Suppressor"]]
        )

        reflData[REFL_COLUMN_NAMES["R Err"]] = np.sqrt(
            reflData[REFL_COLUMN_NAMES["R"]]
            / (
                metaData[REFL_COLUMN_NAMES["Beam Current"]]
                * metaData[REFL_COLUMN_NAMES["Higher Order Suppressor"]]
            )
        )

        reflData[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflData], axis=1)
        return df

    @staticmethod
    def getNormal(reflDataFrame: pd.DataFrame) -> pd.DataFrame:
        _izero_count = 0
        while True:
            if reflDataFrame[REFL_COLUMN_NAMES["Q"]][_izero_count] == 0:
                _izero_count += 1
            else:
                break
        if _izero_count > 0:
            izero: float = np.average(reflDataFrame[REFL_COLUMN_NAMES["R"]].iloc[: _izero_count - 1])  # type: ignore
            izero_err_avg = np.average(
                reflDataFrame[REFL_COLUMN_NAMES["R Err"]].iloc[: _izero_count - 1]
            )
            izero_err: float = izero_err_avg / np.sqrt(_izero_count)
        else:
            izero, izero_err = (1, 1)

        reflDataFrame[REFL_COLUMN_NAMES["R"]] = reflDataFrame[REFL_COLUMN_NAMES["R"]] / izero  # type: ignore
        reflDataFrame[REFL_COLUMN_NAMES["R Err"]] = reflDataFrame[
            REFL_COLUMN_NAMES["R Err"]
        ] * np.sqrt(
            (
                (reflDataFrame[REFL_COLUMN_NAMES["R Err"]])
                / (reflDataFrame[REFL_COLUMN_NAMES["R Err"]])
            )
            ** 2
            + (izero_err / izero)  # type: ignore
        )

        reflDataFrame.drop(reflDataFrame.index[:_izero_count])
        return reflDataFrame

    @staticmethod
    def getOverlaps(col: pd.Series) -> dict:
        tally = defaultdict(list)
        for i, val in col.items():
            tally[val].append(i)
        return {key: val for key, val in tally.items() if len(val) > 1}

    @staticmethod
    def getScaleFactors(
        overlapDict: dict, df: pd.DataFrame, refl=REFL_COLUMN_NAMES["R"]
    ) -> dict:
        reflCol = df[refl]
        keys = list(overlapDict.keys())
        indices = list(overlapDict.values())
        scaleFactors = {}
        for i in range(len(keys)):
            if len(indices[i]) > 2:
                initIndex = indices[i][0]
                overIndices = indices[i][1:]
                initVal = reflCol.iloc[initIndex]
                overVals = reflCol.iloc[overIndices]
                ratio = list(initVal / overVals)
                j = 0
                while len(indices[j]) == 2 and j < len(keys):
                    initVal = reflCol.iloc[indices[j][0]]
                    overVal = reflCol.iloc[indices[j][1]]
                    ratio.append(initVal / overVal)
                    j += 1
                scaleFactors[overIndices[0]] = np.average(ratio)
        return scaleFactors

    @staticmethod
    def replaceWithAverage(df: pd.DataFrame, indices: list) -> pd.DataFrame:
        avg_values = df.mean(axis=1)

        def replace(index):
            df.iloc[index] = avg_values

        with ThreadPoolExecutor() as executor:
            executor.map(replace, indices)

        return df

    @staticmethod
    def scaleSeries(
        df: pd.DataFrame,
        refl: str = REFL_COLUMN_NAMES["R"],
        q: str = REFL_COLUMN_NAMES["Q"],
        reduce: bool = True,
    ) -> pd.DataFrame:
        df = ReflProcs.getNormal(df)
        overlapDict = ReflProcs.getOverlaps(df[q])
        scaleFactors = ReflProcs.getScaleFactors(overlapDict, df, refl=refl)

        for key, val in scaleFactors.items():
            sliceCopy = df.loc[int(key) :, refl].copy()
            df.loc[int(key) :, refl] = val * sliceCopy
        if reduce:
            df = ReflProcs.replaceWithAverage(df, list(scaleFactors.values()))
            for val in overlapDict.values():
                df.drop(val[1:], inplace=True)
        return df
