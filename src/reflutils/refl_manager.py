import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

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
    def addAverageCol(df):
        averageDF = {}
        for col in df.columns:
            if col == REFL_COLUMN_NAMES["R"]:
                average, variance = ErrorManager.weightedAverage(df[col], df[REFL_COLUMN_NAMES["R Err"]])
                averageDF[col] = average
                averageDF[REFL_COLUMN_NAMES["R Err"]] = variance
            elif col == REFL_COLUMN_NAMES["R Err"]:
                pass
            else:
                averageDF[col] = df[col].mean()
        df = pd.concat([df, pd.DataFrame(averageDF, index=[0])], ignore_index=True)
        return df

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
        df[REFL_COLUMN_NAMES["Stat Update"]] = rawIntensityArray.var() / rawIntensityArray.mean()  # type: ignore
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


class ReflFactory:
    @staticmethod
    def getBeamSpots(
        imageList: list, mask: np.ndarray | None = None
    ) -> tuple[pd.DataFrame, list, list]:
        '''This is the main process for generating the beam spots. This takes a list of images and a mask and returns a DataFrame with the images used to calculate the reflectivity as well as the beam and dark spots.'''
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
                "Beam Spots": beamSpots,
                "Dark Spots": darkSpots,
            }
        )
        return images

    @staticmethod
    def getSubImages(
        imageDF: pd.DataFrame,
        height: int = 20,
        width: int = 20,
    ):
        '''This is the main process for generating the sub images. This takes a DataFrame with the images used to calculate the reflectivity as well as the beam and dark spots and returns a DataFrame with the sub images used to calculate the reflectivity.'''
        maskedImages = imageDF[REFL_COLUMN_NAMES["Masked"]]
        beamSpots = imageDF["Beam Spots"]
        darkSpots = imageDF["Dark Spots"]

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
        '''This is the main process for generating the reflectivity. This takes a DataFrame with the sub images used to calculate the reflectivity and returns a DataFrame with the reflectivity.'''
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

        reflDF[REFL_COLUMN_NAMES["R Scale"]] = reflDF[REFL_COLUMN_NAMES["Raw"]] / metaScale

        reflDF[REFL_COLUMN_NAMES["Q"]] = XrayDomainTransform.toQ(
            metaData[REFL_COLUMN_NAMES["Beamline Energy"]],
            metaData[REFL_COLUMN_NAMES["Sample Theta"]],
        )
        df = pd.concat([metaData, reflDF], axis=1)
        df["sf"] = 1
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
        imageDF = ReflFactory.getBeamSpots(imageList, mask=mask)
        imageDF = ReflFactory.getSubImages(
            imageDF, height=height, width=width
        )
        df = ReflFactory.getDf(metaDataFrame, imageDF)
        imageDF[REFL_COLUMN_NAMES["Q"]] = df[REFL_COLUMN_NAMES["Q"]].values
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

        return imageLists, metaDataFrames


class OverlapFactory:
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


class StitchManager:
    @staticmethod
    def getNormal(izeroDataFrame: pd.DataFrame, izeroCount: int):
        # calculates the scale factors
        ErrorManager.updateStats(izeroDataFrame, izeroCount)

        izero = izeroDataFrame[REFL_COLUMN_NAMES["R"]].iat[0]
        izeroVar = izeroDataFrame[REFL_COLUMN_NAMES["R Err"]].iat[0]
        izeroDataFrame["lam"] = izero
        izeroDataFrame["lamErr"] = izeroVar

        izeroDataFrame[REFL_COLUMN_NAMES["R"]] = (
            izeroDataFrame[REFL_COLUMN_NAMES["R"]] / izero
        )

        izeroDataFrame[REFL_COLUMN_NAMES["R Err"]] = (izeroDataFrame[REFL_COLUMN_NAMES["R"]]**2) * ((izeroDataFrame[REFL_COLUMN_NAMES["Stat Update"]] / izeroDataFrame[REFL_COLUMN_NAMES["Raw"]]) + (izeroVar / izero**2))
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


    @staticmethod
    def scaleDataFrame(imageDataFrames, reflDataFrames: list[pd.DataFrame]):
        
        izeroCount, initialOverlapCount, overlaps = OverlapFactory.getOverlap(reflDataFrames[0], reflDataFrames[1])

        for i, df in enumerate(reflDataFrames):
            if i == 0:
                StitchManager.getNormal(df, izeroCount)
            else:
                if i == len(reflDataFrames) - 1:
                    test = 0
                StitchManager.getScaled(df, reflDataFrames[i - 1], initialOverlapCount, overlaps)
        refl = pd.concat(reflDataFrames, ignore_index=True)
        image = pd.concat(imageDataFrames, ignore_index=True)
        return refl, image