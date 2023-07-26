from asyncio import as_completed
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import matplotlib.pyplot as plt

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


class ErrorManager:
    @staticmethod
    def weightedAverage(
        nominal: pd.Series | pd.DataFrame, variance: pd.Series | pd.DataFrame
    ):
        weight = 1 / variance
        variance = weight.sum()
        average = len(weight) * variance
        return average, variance

    @staticmethod
    def dfWeightedAverage(
        df: pd.DataFrame | pd.Series, nominal: str, variance: str
    ) -> tuple:
        """
        This method computed the weighted average and the variance in the weighted average for a DataFrame with one column of nominal values and another of variances. The use of variance here prevents the need to compute square roots at every point along the way of the calculation.
        """
        if nominal not in df.columns:
            raise ValueError(
                f"Invalid column - {nominal} is not a valid columns of the DataFrame"
            )
        if variance not in df.columns:
            raise ValueError(
                f"Invalid column - {variance} is not a valid columns of the DataFrame"
            )
        nom = df[nominal]
        if "k" in df.columns:
            var = df["k"] * df[variance]
        else:
            var = df[variance]
        return ErrorManager.weightedAverage(nom, var)

    @staticmethod
    def updateStats(
        df: pd.DataFrame,
        updatePoints: list[int],
        nominal: str = REFL_COLUMN_NAMES["R"],
        variance: str = REFL_COLUMN_NAMES["R Err"],
        final: int | None = None,
        k="k",
    ):
        """
        Updates the variance in the DataFrame, using the variance across the update points data frame.
        """
        dfSlice = df.loc[updatePoints].copy()
        average, averageVariance = ErrorManager.dfWeightedAverage(
            dfSlice, nominal, variance
        )

        scale = averageVariance / average
        if final == None:
            final = len(df)

        df.loc[updatePoints[0], nominal] = average
        df.loc[updatePoints[0], variance] = averageVariance
        df.loc[updatePoints[0] : final, k] = scale

        df = df.drop(index=updatePoints[0:-2])
        return df

    @staticmethod
    def scaleRefl(
        df: pd.DataFrame,
        scaleFactor: tuple,
        indices: list[int] | None = None,
        refl=REFL_COLUMN_NAMES["R"],
        err=REFL_COLUMN_NAMES["R Err"],
        k="k",
    ):
        scale, scaleErr = scaleFactor
        if indices == None:
            indices = df.index.to_list()

        dfCopy = df.loc[indices]
        df.loc[indices, refl] = dfCopy[refl] / scale
        df.loc[indices, err] = (df.loc[indices, refl] ** 2) * (
            (scaleErr / scale) ** 2 + (dfCopy[k] / dfCopy[refl]) ** 2
        )
        return df


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
        """
        Main process for generating the initial reflectivity DataFrame. This initialized important columns such as the error, spot intensities and other values. This appends to the metaData information collected from the imageDF
        """
        reflDF = ReflProcs.getRefl(imageDF)
        reflDF.reset_index(drop=True, inplace=True)
        
        reflDF[REFL_COLUMN_NAMES["Raw"]] = reflDF[REFL_COLUMN_NAMES["Beam Spot"]] - reflDF[REFL_COLUMN_NAMES["Dark Spot"]]

        metaScale = (metaData[REFL_COLUMN_NAMES["Beam Current"]] * metaData[REFL_COLUMN_NAMES["EXPOSURE"]])

        reflDF[REFL_COLUMN_NAMES["R"]] = reflDF[REFL_COLUMN_NAMES["Raw"]] / metaScale

        reflDF[REFL_COLUMN_NAMES["R Err"]] = reflDF[REFL_COLUMN_NAMES["R"]] / metaScale ** 2

        reflDF[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflDF], axis=1)
        return df

    @staticmethod
    def getNormal(refl: pd.DataFrame, izeroPoints) -> pd.DataFrame:
        refl['lam'] = 1
        refl['lamErr'] = 1
        refl = ErrorManager.updateStats(refl, izeroPoints)
        izero = (
            refl.loc[izeroPoints[-1], REFL_COLUMN_NAMES["R"]],
            refl.loc[izeroPoints[-1], REFL_COLUMN_NAMES["R Err"]],
        )
        refl = ErrorManager.scaleRefl(refl, scaleFactor=izero)

        refl = refl.drop(refl.index[0]).reset_index(drop=True)

        return refl

    @staticmethod
    def getOverlaps(col: pd.Series) -> tuple[list, ...]:
        """
        overlap indices list constructor. This returns three items

        stichZero: list
            izero like list of indices corresponding to the first stich point.

        initialPoints: list
            points corresponding to elements before the stich that overlap with those that occur after the stich

        stichPoints: list
            points corresponding to elements after the stich that overlap with the initialPoints

        cutoff: list
            indices that mark the beginning of a new stich
        """
        tally = defaultdict(list)
        for i, val in col.items():
            tally[val].append(i)

        overlaps = [val for val in tally.values() if len(val) > 1]

        stitches = overlaps[1:]
        izero = overlaps[0]
        period = ReflProcs.getPeriodicity([len(e) for e in stitches])

        stichZero = stitches[::period]
        initialPoints = [
            [s[0] for s in stitches][i * period : (i + 1) * period]
            for i in range(len(stitches) // period)
        ]
        stichPoints = [
            [s[-1] for s in stitches][i * period : (i + 1) * period]
            for i in range(len(stitches) // period)
        ]
        stichSlices = [
            (stichZero[i][1], stichZero[i + 1][1]) for i in range(0, len(stichZero) - 1)
        ]
        return izero, stichZero, initialPoints, stichPoints, stichSlices

    @staticmethod
    def getPrefixArray(pattern):
        """
        Helper function for the KMP
        """
        m = len(pattern)
        prefix = [0] * m
        j = 0
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = prefix[j - 1]

            if pattern[i] == pattern[j]:
                j += 1

            prefix[i] = j
        return prefix

    @staticmethod
    def getPeriodicity(list):
        """
        Implementation of the KPM periodicity finding algorithm This is used to reduce the number of instructions needed for stitching.
        """
        n = len(list)
        prefix = ReflProcs.getPrefixArray(list)

        period = n - prefix[-1]
        if n % period != 0:
            raise ValueError(f"Invalid overlap list - {list} is not periodic")
        return period

    @staticmethod
    def stitchUpdateStats(
        df: pd.DataFrame,
        izero,
        stichZero,
        initialPoints,
        stichPoints,
        stichSlices,
        refl=REFL_COLUMN_NAMES["R"],
        k="k",
    ):
        """
        Updates the k scale at each stich point
        """
        dfNormal = ReflProcs.getNormal(df, izero)
        dfChunks = [dfNormal[slice[0] : slice[1]] for slice in stichSlices]
        result = [dfNormal[0:stichSlices[0][0]]] #not ideal

        for i, (chunk, idx) in enumerate(zip(dfChunks, stichZero)):
            # Handles all other procs
            idx = idx[1:]
            dfUpdated = ErrorManager.updateStats(chunk, idx)
            lastChunk = result[i - 1]

            S = dfUpdated.loc[stichPoints[i - 1], refl]  # stitch intensity
            ks = dfUpdated.loc[stichPoints[i - 1], k]  # err scale stitch
            R = lastChunk.loc[initialPoints[i - 1], refl]  # refl intensity
            kr = lastChunk.loc[initialPoints[i - 1], k]  # err scale refl

            lams = S / R
            lamsErr = lams**2 * (ks / S + kr / R)

            scale = ErrorManager.weightedAverage(lams, lamsErr)
            dfUpdated["lam"] = [scale[0]] * len(dfUpdated)
            dfUpdated["lamErr"] = [scale[1]] * len(dfUpdated)

            refl = ErrorManager.scaleRefl(dfUpdated, scale)
            result.append(ErrorManager.updateStats(chunk, idx))
        refl = pd.concat(result)
        return refl

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
        overlaps = ReflProcs.getOverlaps(df[q])
        scaleFactors = ReflProcs.stitchUpdateStats(df, *overlaps)

        for key, val in scaleFactors.items():
            ratio, ratioErr = val
            sliceCopy = df.loc[int(key) :, [refl, refl_err]].copy()
            df.loc[int(key) :, refl] = ratio * sliceCopy[refl]
            df.loc[int(key) :, refl_err] = sliceCopy[refl] * np.sqrt(
                (sliceCopy[refl_err] / sliceCopy[refl]) ** 2 + (ratioErr / ratio) ** 2
            )

        if reduce:
            df = ReflProcs.replaceWithAverage(df, list(scaleFactors.values()))
            for val in overlapDict.values():
                df.drop(val[1:], inplace=True)
        return df
