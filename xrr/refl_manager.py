from typing import Literal
from unicodedata import numeric
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from xrr._config import REFL_COLUMN_NAMES
    from xrr.image_manager import ImageProcs
    from xrr.toolkit import XrayDomainTransform
    from xrr.refl_reuse import Reuse
except:
    from _config import REFL_COLUMN_NAMES
    from image_manager import ImageProcs
    from toolkit import XrayDomainTransform
    from refl_reuse import Reuse


class ReflProcs:
    @staticmethod
    def getBeamSpots(
        imageList: list, mask: np.ndarray | None = None
    ) -> tuple[pd.DataFrame, list, list]:
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
        images = pd.DataFrame(
            {
                REFL_COLUMN_NAMES["Images"]: imageList,
                REFL_COLUMN_NAMES["Masked"]: maskedImages,
                REFL_COLUMN_NAMES["Filtered"]: filteredImages,
            }
        )
        return images, beamSpots, darkSpots

    @staticmethod
    def getSubImages(
        imageDF: pd.DataFrame,
        beamSpots: list[tuple],
        darkSpots: list[tuple],
        height: int = 20,
        width: int = 20,
    ):
        maskedImages = imageDF[REFL_COLUMN_NAMES["Masked"]]
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
        imageDF[REFL_COLUMN_NAMES["Beam Image"]] = directBeam
        imageDF[REFL_COLUMN_NAMES["Dark Image"]] = backgroundNoise

        return imageDF

    @staticmethod
    def getRefl(imageDF: pd.DataFrame) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            intensity = list(
                executor.map(
                    lambda image: ImageProcs.sumImage(image),
                    imageDF[REFL_COLUMN_NAMES["Beam Image"]],
                )
            )
            background = list(
                executor.map(
                    lambda image: ImageProcs.sumImage(image),
                    imageDF[REFL_COLUMN_NAMES["Dark Image"]],
                )
            )
        reflDF = pd.DataFrame(
            {
                REFL_COLUMN_NAMES["Beam Spot"]: intensity,
                REFL_COLUMN_NAMES["Dark Spot"]: background,
            }
        )
        return reflDF

    @staticmethod
    def getDf(metaData: pd.DataFrame, imageDF) -> pd.DataFrame:
        reflDF = ReflProcs.getRefl(imageDF)
        reflDF.reset_index(drop=True, inplace=True)

        reflDF[REFL_COLUMN_NAMES["R"]] = (
            reflDF[REFL_COLUMN_NAMES["Beam Spot"]]
            - reflDF[REFL_COLUMN_NAMES["Dark Spot"]]
        ) / (
            metaData[REFL_COLUMN_NAMES["Beam Current"]]
            * metaData[REFL_COLUMN_NAMES["Higher Order Suppressor"]]
        )
        reflDF[REFL_COLUMN_NAMES["Raw"]] = reflDF[REFL_COLUMN_NAMES["R"]]

        reflDF[REFL_COLUMN_NAMES["R Err"]] = (
            np.sqrt(reflDF[REFL_COLUMN_NAMES["R"]])
            / metaData[REFL_COLUMN_NAMES["Beam Current"]]
            / metaData[REFL_COLUMN_NAMES["Higher Order Suppressor"]]
        )

        reflDF[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflDF], axis=1)
        return df

    @staticmethod
    def getNormal(reflDataFrame: pd.DataFrame) -> pd.DataFrame:
        izero_count = (reflDataFrame[REFL_COLUMN_NAMES["Q"]] != 0).argmax()

        if izero_count > 0:
            izero: float = np.average(
                reflDataFrame[REFL_COLUMN_NAMES["R"]].iloc[: izero_count],
                weights=1 / (reflDataFrame[REFL_COLUMN_NAMES["R Err"]].iloc[: izero_count])**2
            )  # type: ignore

            izero_err = np.average(
                reflDataFrame[REFL_COLUMN_NAMES["R Err"]].iloc[: izero_count]**2
            )
        else:
            izero, izero_err = (1, 1)

        reflDataFrame[REFL_COLUMN_NAMES["R"]] = (
            reflDataFrame[REFL_COLUMN_NAMES["R"]] / izero
        )

        reflDataFrame[REFL_COLUMN_NAMES["R Err"]] = reflDataFrame[
            REFL_COLUMN_NAMES["R Err"]
        ] * np.sqrt(
            (
                (reflDataFrame[REFL_COLUMN_NAMES["R Err"]])
                / (reflDataFrame[REFL_COLUMN_NAMES["R Err"]])
            )
            ** 2
            + (izero_err / izero) ** 2
        )

        reflDataFrame = reflDataFrame[izero_count:].reset_index(drop=True)
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
                scaleFactors[overIndices[0]] = (
                    np.average(
                        ratio,
                        weights=1
                        / (df[REFL_COLUMN_NAMES["R Err"]].iloc[overIndices]) ** 2,
                    ),
                    np.average(df[REFL_COLUMN_NAMES["R Err"]].iloc[overIndices]**2)
                )
        return scaleFactors

    @staticmethod
    def replaceWithAverage(df: pd.DataFrame, indices: list) -> pd.DataFrame:
        avg_values = df.mean(axis=1)

        def replace(index):
            df.iloc[index, df.columns] = avg_values

        with ThreadPoolExecutor() as executor:
            executor.map(replace, indices)

        return df

    @staticmethod
    def scaleSeries(
        df: pd.DataFrame,
        refl: str = REFL_COLUMN_NAMES["R"],
        q: str = REFL_COLUMN_NAMES["Q"],
        refl_err: str = REFL_COLUMN_NAMES["R Err"],
        reduce: bool = True,
    ) -> pd.DataFrame:
        df = ReflProcs.getNormal(df)
        overlapDict = ReflProcs.getOverlaps(df[q])
        scaleFactors = ReflProcs.getScaleFactors(overlapDict, df, refl=refl)

        for key, val in scaleFactors.items():
            ratio, ratioErr = val
            sliceCopy = df.loc[int(key) :, [refl, refl_err]].copy()
            df.loc[int(key) :,refl] = ratio * sliceCopy[refl]
            df.loc[int(key) :,refl_err] = sliceCopy[refl] * np.sqrt(
                (sliceCopy[refl_err] / sliceCopy[refl]) ** 2 + (ratioErr / ratio) ** 2
            )

        if reduce:
            df = ReflProcs.replaceWithAverage(df, list(scaleFactors.values()))
            for val in overlapDict.values():
                df.drop(val[1:], inplace=True)
        return df
