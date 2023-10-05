import numpy as np
import pandas as pd
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
    def addAverageCol(df: pd.DataFrame, averagePoints, inPlace: bool = True):
        """Adds a row to the dataframe with the averages of the columns"""
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
        average = pd.DataFrame(averageDF, index=[averagePoints - 1])
        df_pre = df.iloc[:averagePoints]
        df_pre.index = [0]*averagePoints
        df_post = df.iloc[averagePoints:]
        df = pd.concat([df_pre, average, df_post])

        if not inPlace:
            return df

    @staticmethod
    def updateStats(
        df: pd.DataFrame,
        updatePoints: int,
        raw: str = REFL_COLUMN_NAMES["Raw"],
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
        inPlace: bool = True,
    ):
        if REFL_COLUMN_NAMES["i0"] not in df.columns:
            df[REFL_COLUMN_NAMES["i0"]] = df[REFL_COLUMN_NAMES["Beam Current"]]
            df[REFL_COLUMN_NAMES["i0Err"]] = 0

        df[REFL_COLUMN_NAMES["R"]] = df[REFL_COLUMN_NAMES["R Raw"]] / df[REFL_COLUMN_NAMES["i0"]]
        df[REFL_COLUMN_NAMES["R Err"]] = df[REFL_COLUMN_NAMES["Stat Update"]]*df[REFL_COLUMN_NAMES["R"]]

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
    def normalize(izeroDataFrame: pd.DataFrame, izeroCount: int):
        """Normalizes the first stitch dataframe"""

        # calculates the i0 collumn information
        izero = izeroDataFrame[REFL_COLUMN_NAMES["Raw"]].iloc[:izeroCount].mean()
        izeroVar = izeroDataFrame[REFL_COLUMN_NAMES["Raw"]].iloc[:izeroCount].var()
        izeroDataFrame[REFL_COLUMN_NAMES["i0"]] = izero
        izeroDataFrame[REFL_COLUMN_NAMES["i0Err"]] = izeroVar

        # calculates the reflectivity collumn information
        izeroDataFrame[REFL_COLUMN_NAMES["R"]] = izeroDataFrame[REFL_COLUMN_NAMES["Raw"]] / izero
        izeroDataFrame[REFL_COLUMN_NAMES["R Err"]] = izeroDataFrame[REFL_COLUMN_NAMES["Stat Update"]]*izeroDataFrame[REFL_COLUMN_NAMES["R"]]

    @staticmethod
    def scaleFactor(currentStitchPoints, PriorStitchPoints):
        """Calculates the scale factor for the current stitch dataframe"""
        scaleFactors = PriorStitchPoints[REFL_COLUMN_NAMES["Raw"]] / currentStitchPoints[REFL_COLUMN_NAMES["Raw"]]
        scaleVars = scaleFactors ** 2 * (
            PriorStitchPoints[REFL_COLUMN_NAMES["Stat Update"]]
            / PriorStitchPoints[REFL_COLUMN_NAMES["Raw"]]
            + currentStitchPoints[REFL_COLUMN_NAMES["Stat Update"]]
            / currentStitchPoints[REFL_COLUMN_NAMES["Raw"]]
        )

        scaleFactor, scaleVar = ErrorManager.weightedAverage(scaleFactors, scaleVars)
        return scaleFactor, scaleVar

    @staticmethod
    def stitch(
        currentDataFrame: pd.DataFrame,
        priorDataFrame: pd.DataFrame,
        initialOverlapCount: int,
        overlaps: int,
    ):
        '''Scales future stitch dataframes to their prior stitch dataframes'''
        # deal with the inital overlap point
        ErrorManager.addAverageCol(currentDataFrame, initialOverlapCount)

        # slice dataframes into the overlap region
        currentStitchPoints = currentDataFrame.iloc[overlaps-1:2*overlaps]
        priorStitchPoints = priorDataFrame.iloc[-overlaps:]

       
        scale, scaleVar = StitchManager.scaleFactor(currentStitchPoints, priorStitchPoints)
        # calculate the scale factor
        currentDataFrame[REFL_COLUMN_NAMES["i0"]] = scale
        currentDataFrame[REFL_COLUMN_NAMES["i0Err"]] = scaleVar 

        currentDataFrame[REFL_COLUMN_NAMES["R"]] = (
            currentDataFrame[REFL_COLUMN_NAMES["Raw"]] * currentDataFrame[REFL_COLUMN_NAMES["i0"]]
        )

        currentDataFrame[REFL_COLUMN_NAMES["R Err"]] = currentDataFrame[REFL_COLUMN_NAMES["R"]]**2 * (
            currentDataFrame[REFL_COLUMN_NAMES["Stat Update"]] / currentDataFrame[REFL_COLUMN_NAMES["Raw"]]
            + currentDataFrame[REFL_COLUMN_NAMES["i0Err"]] / priorDataFrame[REFL_COLUMN_NAMES["i0"]]**2
        )


    @staticmethod
    def scaleDataFrame(imageDataFrames, reflDataFrames: list[pd.DataFrame]):
        
        izeroCount, initialOverlapCount, overlaps = OverlapFactory.getOverlap(reflDataFrames[0], reflDataFrames[1])

        for i, df in enumerate(reflDataFrames):
            ErrorManager.updateStats(df, izeroCount)
            if i == 0:
                StitchManager.normalize(df, izeroCount)
            else:
                if i == len(reflDataFrames) - 1:
                    test = 0
                StitchManager.stitch(df, reflDataFrames[i - 1], initialOverlapCount, overlaps)
        refl = pd.concat(reflDataFrames, ignore_index=True)
        image = pd.concat(imageDataFrames, ignore_index=True)
        return refl, image