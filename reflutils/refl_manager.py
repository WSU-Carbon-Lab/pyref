import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from warnings import warn

import matplotlib.pyplot as plt

try:
    from reflutils._config import REFL_COLUMN_NAMES
    from reflutils.image_manager import ImageProcs
    from reflutils.toolkit import XrayDomainTransform
except:
    from _config import REFL_COLUMN_NAMES
    from image_manager import ImageProcs
    from toolkit import XrayDomainTransform


class ErrorManager:
    @staticmethod
    def weightedAverage(
        nominal: pd.Series | pd.DataFrame | np.ndarray, variance: pd.Series | pd.DataFrame|np.ndarray
    ):
        """
        Computes the weighted average of a series of nominal points and their variance
        """

        weight = 1 / variance
        variance = 1 / weight.sum()
        average = (nominal * weight).sum() * variance

        return average, variance

    @staticmethod
    def updateStats(
        df: pd.DataFrame,
        updatePoints: list[int],
        raw: str = REFL_COLUMN_NAMES["Raw"],
        k: str = "k",
        inPlace: bool = True,
    ):
        """
        Computes the ADU to photon conversion factor
        """
        rawIntensityArray = df.loc[updatePoints, raw]
        df[k] = rawIntensityArray.var() / rawIntensityArray.mean()  # type: ignore
        if not inPlace:
            return df

    @staticmethod
    def scaleRefl(
        df: pd.DataFrame,
        scaleCol: str,
        scaleVarCol: str,
        nominalCol: str = REFL_COLUMN_NAMES["R"],
        varCol=REFL_COLUMN_NAMES["R Err"],
        inPlace: bool = True,
    ):
        if scaleCol not in df.columns:
            df[scaleCol] = 1

        df[varCol] = (df[nominalCol] / df[scaleCol]) ** 2 * (
            (df[varCol] / df[nominalCol]) ** 2 + (df[scaleVarCol] / df[scaleCol]) ** 2
        )
        df[nominalCol] = df[nominalCol] / df[scaleCol]

        if not inPlace:
            return df

    @staticmethod
    def overloadMean(
        df: pd.DataFrame,
        meanIndexes: list[int],
        refl=REFL_COLUMN_NAMES["R"],
        var=REFL_COLUMN_NAMES["R Err"],
        inplace=True,
    ):
        """
        Replace all meanIndexes points with the average across them
        """
        for col in df.columns:
            if col == refl:
                (
                    df.loc[meanIndexes, col],
                    df.loc[meanIndexes, var],
                ) = ErrorManager.weightedAverage(
                    df.loc[meanIndexes, col], df.loc[meanIndexes, var]
                )
            else:
                df.loc[meanIndexes, col] = df.loc[meanIndexes, col].mean()
        df.drop(index=meanIndexes[:-1], inplace=True)

        if not inplace:
            return df

    @staticmethod
    def getScaleFactor(currentDataFrame: pd.DataFrame, priorDataFrame: pd.DataFrame):
        scaleFactors = (
            currentDataFrame[REFL_COLUMN_NAMES["R"]].values
            / priorDataFrame[REFL_COLUMN_NAMES["R"]].values
        )
        scaleVars = scaleFactors**2 * (
            currentDataFrame["k"].values
            / currentDataFrame[REFL_COLUMN_NAMES["Raw"]].values
            + priorDataFrame["k"].values
            / priorDataFrame[REFL_COLUMN_NAMES["Raw"]].values
        )
        scaleFactor, scaleVar = ErrorManager.weightedAverage(scaleFactors, scaleVars)
        if 1 / scaleFactor > 1:
            outlierPoints = [scale for scale in scaleFactors if scale > 1]
            warn(f"One or more stitch points is an outlier - {outlierPoints} are too large")
            scaleFactors = np.array([scale for scale in scaleFactors if scale > 1])
            scaleVars =  np.array([var for scale, var in zip(scaleFactors, scaleVars) if scale > 1])

            scaleFactor, scaleVar = ErrorManager.weightedAverage(scaleFactors, scaleVars)

        return scaleFactor, scaleVar

    @staticmethod
    def toStd(stitchedDataFrame):
        '''Manages conversions from the variance used in the calculations to the standard error that is standard.'''
        stitchedDataFrame[REFL_COLUMN_NAMES["R Err"]] = np.sqrt(stitchedDataFrame[REFL_COLUMN_NAMES["R Err"]])


class ReflFactory:
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
        reflDF = ReflFactory.getRefl(imageDF)
        reflDF.reset_index(drop=True, inplace=True)

        reflDF[REFL_COLUMN_NAMES["Raw"]] = (
            reflDF[REFL_COLUMN_NAMES["Beam Spot"]]
            - reflDF[REFL_COLUMN_NAMES["Dark Spot"]]
        )

        metaScale = (
            metaData[REFL_COLUMN_NAMES["Beam Current"]]
            # * metaData[REFL_COLUMN_NAMES["EXPOSURE"]]
        )

        reflDF[REFL_COLUMN_NAMES["R"]] = reflDF[REFL_COLUMN_NAMES["Raw"]] / metaScale

        reflDF[REFL_COLUMN_NAMES["R Err"]] = reflDF[REFL_COLUMN_NAMES["R"]] / metaScale

        reflDF[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflDF], axis=1)
        return df


class OverlapFactory:
    @staticmethod
    def getOverlaps(col: pd.Series) -> tuple:
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
        period = OverlapFactory.getPeriodicity([len(e) for e in stitches])

        stitches = overlaps[1:]
        izero = overlaps[0]

        stichZero = [izero] + stitches[::period]
        seriesSlices = []
        seriesN = len(stichZero)
        for i in range(1, seriesN):
            if i == seriesN - 1:
                seriesSlices.append(slice(stichZero[i][1], len(col)))
            else:
                seriesSlices.append(slice(stichZero[i][1], stichZero[i + 1][1]))
        seriesSlices = [slice(0, stichZero[1][1])] + seriesSlices
        return stichZero, seriesSlices, period, overlaps

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
        prefix = OverlapFactory.getPrefixArray(list)

        period = n - prefix[-1]
        if n % period != 0:
            raise ValueError(f"Invalid overlap list - {list} is not periodic")
        if period == n:
            raise ValueError(f"Invalid overlap list - {list} is not periodic")
            
        return period


class OutlierDetection:
    ''' 
    Wrapper class for removing outliers from the dataset.
    '''

    @staticmethod
    def visualizeDataPoints(beamSpots, imageSize, *args, **kwargs):
        X = np.array(beamSpots)

        model = OutlierDetection.isolationForest(X)

        disp = DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            response_method="decision_function",
            alpha = .5,
        )

        disp.ax_.scatter(X[:,0], X[:,1], s = 20)
        disp.ax_.set_xlabel("Pixel Index")
        disp.ax_.set_ylabel("Pixel Index")

        plt.title("Path Length decision boundary \nof Isolation Forest Algorithm")
        plt.gca().invert_yaxis()
        plt.legend(labels = ["Brightest CCD \nPixel Location"])
        plt.show()


    
    @staticmethod
    def isolationForest(X, *args, **kwargs):
        model = IsolationForest(max_samples=100, bootstrap=True, *args, **kwargs)
        model.fit(X)
        return model

class StitchManager:
    @staticmethod
    def getNormal(izeroDataFrame: pd.DataFrame, izeroPoints):
        # calculates the scale factors
        ErrorManager.updateStats(izeroDataFrame, izeroPoints)
        ErrorManager.overloadMean(izeroDataFrame, izeroPoints)

        izero = izeroDataFrame.loc[izeroPoints[-1], REFL_COLUMN_NAMES["R"]]
        izeroErr = izeroDataFrame.loc[izeroPoints[-1], REFL_COLUMN_NAMES["R Err"]]
        izeroDataFrame["i0"] = izero
        izeroDataFrame["i0Err"] = izeroErr
        izeroDataFrame[REFL_COLUMN_NAMES["R"]] = (
            izeroDataFrame[REFL_COLUMN_NAMES["R"]] / izero
        )

        izeroDataFrame[REFL_COLUMN_NAMES["R Err"]] = (
            izeroDataFrame[REFL_COLUMN_NAMES["R"]] **2 *(
            izeroDataFrame.loc[izeroPoints[-1], 'k'] / izeroDataFrame.loc[izeroPoints[-1], REFL_COLUMN_NAMES["Raw"]]
            + izeroDataFrame["k"] / izeroDataFrame[REFL_COLUMN_NAMES["Raw"]]
        ))
        izeroDataFrame.drop(izeroPoints[-1], inplace=True)

    @staticmethod
    def getScaled(
        currentDataFrame: pd.DataFrame,
        priorDataFrame: pd.DataFrame,
        stitchZero: list,
        period: int,
    ):
        # calculates the scale factor
        ErrorManager.updateStats(currentDataFrame, stitchZero)

        currentStitchPoints = currentDataFrame[:period]
        priorStitchPoints = priorDataFrame[-period:]

        scale, scaleVar = ErrorManager.getScaleFactor(
            currentStitchPoints, priorStitchPoints
        )
        currentDataFrame["Lam"] = scale
        currentDataFrame["LamErr"] = scaleVar
        currentDataFrame[REFL_COLUMN_NAMES["R"]] = (
            currentDataFrame[REFL_COLUMN_NAMES["R"]] / scale
        )

        currentDataFrame[REFL_COLUMN_NAMES["R Err"]] = (
            (currentDataFrame[REFL_COLUMN_NAMES["R"]])**2 * (
            currentDataFrame["LamErr"] / (currentDataFrame["Lam"])**2 
            + currentDataFrame["k"] / currentDataFrame[REFL_COLUMN_NAMES["Raw"]])
        )
    @staticmethod
    def dropReplace(stitchedDataFrame: pd.DataFrame, overlaps):
        for i, idxList in enumerate(overlaps):
            if i > 0:
                meanValues = stitchedDataFrame.loc[idxList].groupby(level=0).agg("mean")

                stitchedDataFrame.loc[idxList] = meanValues
                stitchedDataFrame.drop(idxList[1:], inplace=True)

    @staticmethod
    def scaleDataFrame(reflDataFrame):
        stitchZero, seriesSlices, period, overlaps = OverlapFactory.getOverlaps(
            reflDataFrame[REFL_COLUMN_NAMES["Sample Theta"]]
        )
        DataFrames = [reflDataFrame[s] for s in seriesSlices]
        stitchedDataFrames = []
        for i, df in enumerate(DataFrames):
            if i == 0:
                StitchManager.getNormal(df, stitchZero[i])
            else:
                StitchManager.getScaled(
                    df, stitchedDataFrames[i - 1], stitchZero[i][1:], period
                )
            stitchedDataFrames.append(df)

        stitchedDataFrame = pd.concat(stitchedDataFrames)

        StitchManager.dropReplace(stitchedDataFrame, overlaps)
        ErrorManager.toStd(stitchedDataFrame)
        return stitchedDataFrame
