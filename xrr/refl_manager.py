from sys import prefix
from typing import Literal
from unicodedata import numeric
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    def propagateVariance(
        df: pd.DataFrame,
        measurementCol: str,
        varianceCol: str,
        scale: float,
        scaleVariance: float,
    ) -> pd.DataFrame:
        cols = df.columns
        if measurementCol not in df.columns:
            raise ValueError(
                f"Invalid Initial Value - {measurementCol} not valid columns in the DataFrame"
            )

        if varianceCol not in df.columns:
            raise ValueError(
                f"Invalid Initial Variance - {varianceCol} not valid columns in the DataFrame"
            )
        dfCopy = df.copy()
        measurement = dfCopy[measurementCol]
        measurementVariance = dfCopy[varianceCol]
        
        dfCopy[varianceCol] = 
        dfCopy[measurementCol] = measurement * scale
        

        return df

    @staticmethod
    def weightedAverage(
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
        weight = 1 / df[variance]

        average = (df[nominal] * weight).sum() / weight.sum()
        variance = 1 / weight.sum()
        return average, variance

    @staticmethod
    def updateStats(
        df: pd.DataFrame, updatePoints: list[int], nominal: str, variance: str, final: int | None = None
    ):
        """
        This allows error handling for single ADU measurements from BL 11.0.1.2. This is our method for converting the ADU to photons. Additionally, the Normalization could occur while the currents are being
        """
        dfSlice = df.iloc[updatePoints].copy()
        average, averageVariance = ErrorManager.weightedAverage(
            dfSlice, nominal, variance
        )

        scale = averageVariance / average
        if final == None:
            final = len(df)
        
        df.loc[updatePoints[0], nominal] = average
        df.loc[updatePoints[0], variance] = averageVariance
        df.loc[updatePoints[1] : final, variance] *= scale
        
        df = df.drop(index=updatePoints[1:-1]).reset_index(drop=True)
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

        reflDF[REFL_COLUMN_NAMES["R"]] = (
            reflDF[REFL_COLUMN_NAMES["Beam Spot"]]
            - reflDF[REFL_COLUMN_NAMES["Dark Spot"]]
        ) / metaData[REFL_COLUMN_NAMES["Beam Current"]]

        reflDF[REFL_COLUMN_NAMES["Raw"]] = reflDF[REFL_COLUMN_NAMES["R"]]

        reflDF[REFL_COLUMN_NAMES["R Err"]] = reflDF[REFL_COLUMN_NAMES["R"]] / metaData[REFL_COLUMN_NAMES["Beam Current"]]

        reflDF[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflDF], axis=1)
        return df

    @staticmethod
    def getNormal(reflDataFrame: pd.DataFrame) -> pd.DataFrame:
        """
        This computes the reflectivity intensity this is computed as

            R = I / I_0

        Where I_0 is the direct beam intensity. An important aspect of this computation is that the I_0 measurement is the intensity of the direct beam. Thus this intensity changes as a function of time depending on the beam Current. So the real computation is

            R = (I / J) / (I_0 / J_0)

        In reality these are computations done discretely and all depend on the current during the direct beam scans.
        """
        izeroPoints = []
        for i, val in reflDataFrame[REFL_COLUMN_NAMES["Q"]].items():
            izeroPoints.append(i)
            if val > 0:
                break

        if len(izeroPoints) > 0:
            reflDataFrame = ErrorManager.updateStats(
                reflDataFrame,
                updatePoints=izeroPoints,
                nominal=REFL_COLUMN_NAMES["R"],
                variance=REFL_COLUMN_NAMES["R Err"],
            )
        izeroScale = 1 / reflDataFrame[REFL_COLUMN_NAMES["R"]].iloc[0]
        izeroScaleErr = reflDataFrame[REFL_COLUMN_NAMES["R Err"]].iloc[0]

        reflDataFrame = ErrorManager.propagateVariance(reflDataFrame, measurementCol=REFL_COLUMN_NAMES["R"], varianceCol=REFL_COLUMN_NAMES["R Err"], scale=izeroScale, scaleVariance=izeroScaleErr)
        reflDataFrame.drop(reflDataFrame.index[0]).reset_index(drop=True)
        return reflDataFrame

    @staticmethod
    def getOverlaps(col: pd.Series) -> list:
        tally = defaultdict(list)
        for i, val in col.items():
            tally[val].append(i)
        return [val for val in tally.values() if len(val) > 1]

    @staticmethod
    def getPrefixArray(pattern):
        m = len(pattern)
        prefix = [0]*m
        j = 0
        for i in range(1, m):
            while j>0 and pattern[i] != pattern[j]:
                j = prefix[j - 1]
            
            if pattern[i] == pattern[j]:
                j+=1
            
            prefix[i] = j
        return prefix

    @staticmethod
    def getPeridicity(list):
        n = len(list)
        prefix = ReflProcs.getPrefixArray(list)

        period = n - prefix[-1]
        if n % period != 0:
            raise ValueError(f'Invalid overlap list - {list} is not periodic')
        return period

    @staticmethod
    def overlapPointRepackage(overlaps: dict):
        
        lengths = [len(over) for over in overlaps]
        period = ReflProcs.getPeridicity(lengths)
        initialStitchPoint = overlaps[0::period]
        stitchPoints = [overlaps[i * period + 1:(i+1)*period] for i in range(len(overlaps)//period)]
            
        return initialStitchPoint, stitchPoints

    @staticmethod
    def findStitchFactor(df: pd.DataFrame, initialStitchPoints: list, stritchPoints: list):
        initPoint = initialStitchPoints[0]
        df
        

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
            df.loc[int(key) :, refl] = ratio * sliceCopy[refl]
            df.loc[int(key) :, refl_err] = sliceCopy[refl] * np.sqrt(
                (sliceCopy[refl_err] / sliceCopy[refl]) ** 2 + (ratioErr / ratio) ** 2
            )

        if reduce:
            df = ReflProcs.replaceWithAverage(df, list(scaleFactors.values()))
            for val in overlapDict.values():
                df.drop(val[1:], inplace=True)
        return df
