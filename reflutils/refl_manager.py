from email.mime import image
import re
from turtle import heading, width
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
        nominal: pd.Series | pd.DataFrame | np.ndarray,
        variance: pd.Series | pd.DataFrame | np.ndarray,
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
        updatePoints: int,
        raw: str = REFL_COLUMN_NAMES["Raw"],
        k: str = "k",
        inPlace: bool = True,
    ):
        """
        Computes the ADU to photon conversion factor
        """
        rawIntensityArray = df[raw][:updatePoints]
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
        meanCutoff: int,
        refl=REFL_COLUMN_NAMES["R"],
        var=REFL_COLUMN_NAMES["R Err"],
        inplace=True,
    ):
        """
        Replace all meanIndexes points with the average across them
        """
        if refl in df.columns:
            weightedAverage, weightedVariance = ErrorManager.weightedAverage(
                df[refl][:meanCutoff], df[var][:meanCutoff]
            )
        else:
            raise ValueError(f"{refl} not in dataframe")
        
        cols = df.drop(columns=[refl, var]).columns
        mean = df.copy()[cols].loc[:meanCutoff].mean().round(4)
        for col in cols:
            df.loc[:,col].iloc[:meanCutoff] = mean[col]
        
        df[refl].iloc[:meanCutoff] = weightedAverage
        df[var].iloc[:meanCutoff] = weightedVariance

        df.drop(df.index[:meanCutoff -1], inplace=True)
        df.reset_index(drop=True, inplace=True)

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
            warn(
                f"One or more stitch points is an outlier - {outlierPoints} are too large"
            )
            scaleFactors = np.array([scale for scale in scaleFactors if scale > 1])
            scaleVars = np.array(
                [var for scale, var in zip(scaleFactors, scaleVars) if scale > 1]
            )

            scaleFactor, scaleVar = ErrorManager.weightedAverage(
                scaleFactors, scaleVars
            )

        return scaleFactor, scaleVar

    @staticmethod
    def toStd(stitchedDataFrame):
        """Manages conversions from the variance used in the calculations to the standard error that is standard."""
        stitchedDataFrame[REFL_COLUMN_NAMES["R Err"]] = np.sqrt(
            stitchedDataFrame[REFL_COLUMN_NAMES["R Err"]]
        )


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
        imageDF.reset_index(drop=True, inplace=True)
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
        return reflDF.reset_index(drop=True, inplace=False)

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

    @staticmethod
    def __main(
        imageList: list,
        metaDataFrame: pd.DataFrame,
        mask: np.ndarray | None = None,
        height: int = 20,
        width: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This is the main process for constructing the reflectivity DataFrame. This takes a list of images, metaData and a mask and returns a DataFrame with the reflectivity information as well as the images used to calculate the reflectivity.
        """
        imageDF, beamSpots, darkSpots = ReflFactory.getBeamSpots(imageList, mask=mask)
        imageDF = ReflFactory.getSubImages(
            imageDF, beamSpots, darkSpots, height=height, width=width
        )
        df = ReflFactory.getDf(metaDataFrame, imageDF)
        return df, imageDF

    @staticmethod
    def main(
        imageLists: list,
        metaDataFrames: list[pd.DataFrame],
        mask: np.ndarray | None = None,
        height=10,
        width=10,
    ):
        for (
            i,
            (imageDF, metaDF),
        ) in enumerate(zip(imageLists, metaDataFrames)):
            df, imageDF = ReflFactory.__main(
                imageDF, metaDF, mask=mask, height=height, width=width
            )
            metaDataFrames[i] = df
            imageLists[i] = imageDF

        return pd.concat(imageLists, ignore_index=True), metaDataFrames


class OverlapFactory:
    @staticmethod
    def DEPRECIATED_getOverlaps(col: pd.Series) -> tuple:
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
    def DEPRECIATED_getPrefixArray(pattern):
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
    def DEPRECIATED_getPeriodicity(list):
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

    @staticmethod
    def getOverlap(initialDataFrame, overlapDataFrame):
        initialPoint = overlapDataFrame[REFL_COLUMN_NAMES["Q"]].iat[0]

        izeroNumber = initialDataFrame[REFL_COLUMN_NAMES["Beam Current"]].where(initialDataFrame[REFL_COLUMN_NAMES["Q"]] == 0).count()

        initialOverlapPoints = overlapDataFrame[REFL_COLUMN_NAMES["Beam Current"]].where(overlapDataFrame[REFL_COLUMN_NAMES["Q"]] == initialPoint).count()
        
        numberOfOverlaps = 1
        for i, q in enumerate(reversed(initialDataFrame[REFL_COLUMN_NAMES["Q"]])):
            if q == overlapDataFrame[REFL_COLUMN_NAMES["Q"]].iat[0]:
                numberOfOverlaps = i
                break
        return izeroNumber, initialOverlapPoints, numberOfOverlaps + 1
    
    
class OutlierDetection:
    """
    Wrapper class for removing outliers from the dataset.
    """

    @staticmethod
    def visualizeDataPoints(beamSpots, imageSize, *args, **kwargs):
        X = np.array(beamSpots)

        model = OutlierDetection.isolationForest(X)

        disp = DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            response_method="decision_function",
            alpha=0.5,
        )

        disp.ax_.scatter(X[:, 0], X[:, 1], s=20)
        disp.ax_.set_xlabel("Pixel Index")
        disp.ax_.set_ylabel("Pixel Index")

        plt.title("Path Length decision boundary \nof Isolation Forest Algorithm")
        plt.gca().invert_yaxis()
        plt.legend(labels=["Brightest CCD \nPixel Location"])
        plt.show()

    @staticmethod
    def isolationForest(X, *args, **kwargs):
        model = IsolationForest(max_samples=100, bootstrap=True, *args, **kwargs)
        model.fit(X)
        return model


class StitchManager:
    @staticmethod
    def getNormal(izeroDataFrame: pd.DataFrame, izeroCount: int):
        # calculates the scale factors
        ErrorManager.updateStats(izeroDataFrame, izeroCount)
        ErrorManager.overloadMean(izeroDataFrame, izeroCount)

        izero = izeroDataFrame[REFL_COLUMN_NAMES["R"]].iat[0]
        izeroVar = izeroDataFrame[REFL_COLUMN_NAMES["R Err"]].iat[0]
        izeroDataFrame["lam"] = izero
        izeroDataFrame["lamErr"] = izeroVar

        izeroDataFrame[REFL_COLUMN_NAMES["R"]] = (
            izeroDataFrame[REFL_COLUMN_NAMES["R"]] / izero
        )

        izeroDataFrame[REFL_COLUMN_NAMES["R Err"]] = (izeroDataFrame[REFL_COLUMN_NAMES["R"]]**2) * ((izeroDataFrame['k'] / izeroDataFrame[REFL_COLUMN_NAMES["Raw"]]) + (izeroVar / izero**2))
        izeroDataFrame.drop(index = 0, inplace=True)

    @staticmethod
    def getScaled(
        currentDataFrame: pd.DataFrame,
        priorDataFrame: pd.DataFrame,
        initialOverlapCount: int,
        overlaps: int,
    ):
        # calculates the scale factor
        ErrorManager.updateStats(currentDataFrame, initialOverlapCount)
        ErrorManager.overloadMean(currentDataFrame, initialOverlapCount)

        currentStitchPoints = currentDataFrame.iloc[:overlaps]
        priorStitchPoints = priorDataFrame.iloc[-overlaps:]

        scale, scaleVar = ErrorManager.getScaleFactor(
            currentStitchPoints, priorStitchPoints
        )
        currentDataFrame["lam"] = scale
        currentDataFrame["lamErr"] = scaleVar
        currentDataFrame[REFL_COLUMN_NAMES["R"]] = (
            currentDataFrame[REFL_COLUMN_NAMES["R"]] / scale
        )

        currentDataFrame[REFL_COLUMN_NAMES["R Err"]] = (
            currentDataFrame[REFL_COLUMN_NAMES["R"]]
        ) ** 2 * (
            currentDataFrame["lamErr"] / (currentDataFrame["lam"]) ** 2
            + currentDataFrame["k"] / currentDataFrame[REFL_COLUMN_NAMES["Raw"]]
        )

        StitchManager.replace(currentDataFrame, priorDataFrame, overlaps)

    @staticmethod
    def replace(currentDataFrame: pd.DataFrame, priorDataFrame: pd.DataFrame, overlaps):
        mean = (currentDataFrame[:overlaps] + priorDataFrame[-overlaps:].values)/2
        currentDataFrame[:overlaps] = mean
        priorDataFrame.drop(priorDataFrame.tail(overlaps).index, inplace=True)


    @staticmethod
    def scaleDataFrame(reflDataFrames: list[pd.DataFrame]):
        
        izeroCount, initialOverlapCount, overlaps = OverlapFactory.getOverlap(reflDataFrames[0], reflDataFrames[1])

        for i, df in enumerate(reflDataFrames):
            if i == 0:
                StitchManager.getNormal(df, izeroCount)
            else:
                if i == len(reflDataFrames) - 1:
                    test = 0
                StitchManager.getScaled(df, reflDataFrames[i - 1], initialOverlapCount, overlaps)
        
        return pd.concat(reflDataFrames, ignore_index=True)



if __name__ == "__main__":
    testInitial = pd.DataFrame({
        "Q": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })
    testOverlap = pd.DataFrame({
        "Q": [7,7, 8, 9, 10, 11, 12, 13, 14, 15],
    })
    testInitial["Q"][:2] = testOverlap["Q"][:2]
    print(testInitial)
    # print(OverlapFactory.getOverlaps(testInitial, testOverlap))