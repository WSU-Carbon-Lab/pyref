from turtle import done
from typing import Any, Literal
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
        variance = 1 / weight.sum(numeric_only=True)
        average = (nominal * weight).sum(numeric_only=True) * variance

        return average, variance
    
    @staticmethod
    def averageOfAbove(df: pd.DataFrame, label = "overlap"):
        """Adds a row to the dataframe with the averages of the columns"""
        averageDF = df.mean(numeric_only=True)
        averageDF[REFL_COLUMN_NAMES["catagory"]] = label
        averageDF[REFL_COLUMN_NAMES["Raw"]]
        averageDF[REFL_COLUMN_NAMES["Stat Update"]]
        return averageDF.to_frame().T

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
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.main(*args, **kwds)

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
            darkSpot = ImageProcs.oppositePoint(beamSpots[0], maskedImages[0].shape)
        images = pd.DataFrame(
            {
                REFL_COLUMN_NAMES["Images"]: imageList,
                REFL_COLUMN_NAMES["Masked"]: maskedImages,
                REFL_COLUMN_NAMES["Filtered"]: filteredImages,
                REFL_COLUMN_NAMES["Beam Spot"]: beamSpots,
            }
        )
        images[REFL_COLUMN_NAMES["Dark Spot"]] = [darkSpot] * len(images)
        return images

    @staticmethod
    def getSubImages(
        imageDF: pd.DataFrame,
        height: int = 20,
        width: int = 20,
    ):
        '''This is the main process for generating the sub images. This takes a DataFrame with the images used to calculate the reflectivity as well as the beam and dark spots and returns a DataFrame with the sub images used to calculate the reflectivity.'''
        maskedImages = imageDF[REFL_COLUMN_NAMES["Masked"]]
        beamSpots = imageDF[REFL_COLUMN_NAMES["Beam Spot"]]
        darkSpots = imageDF[REFL_COLUMN_NAMES["Dark Spot"]]

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
        ) / metaData[REFL_COLUMN_NAMES["Beam Current"]]

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
    def labelDF(df, kind: Literal["i0", "ovp"], izeroCount: int, overlaps: int = 0):
        if kind == 'i0':
            df[REFL_COLUMN_NAMES["catagory"]] = "i0"
            df[REFL_COLUMN_NAMES["catagory"]].iloc[izeroCount:] = "done"
            return df.where(df[REFL_COLUMN_NAMES["catagory"]] == "i0").dropna(), df.where(df[REFL_COLUMN_NAMES["catagory"]] == "done").dropna()
        elif kind == 'ovp':
            df[REFL_COLUMN_NAMES["catagory"]] = "done"
            df[REFL_COLUMN_NAMES["catagory"]].iloc[:izeroCount] = "i0"
            df[REFL_COLUMN_NAMES["catagory"]].iloc[izeroCount:izeroCount+overlaps] = "overlaps"
            i0 = df.where(df[REFL_COLUMN_NAMES["catagory"]] == "i0").dropna()
            overlaps = df.where(df[REFL_COLUMN_NAMES["catagory"]] == "overlaps").dropna()
            done = df.where(df[REFL_COLUMN_NAMES["catagory"]] == "done").dropna()
            return i0, overlaps, done
        

    @staticmethod
    def normalize(izeroDataFrame: pd.DataFrame, izeroCount: int, overlaps = 0, kind: Literal["i0", "ovp"] = "i0"):
        """Normalizes the first stitch dataframe"""
        # catagorize points in the dataframe
        if kind == "i0":
            i0, done = StitchManager.labelDF(izeroDataFrame, kind, izeroCount)
            average = ErrorManager.averageOfAbove(i0, label="done")

            df = pd.concat([i0,average, done], ignore_index=True)
            df[REFL_COLUMN_NAMES["i0"]] = average[REFL_COLUMN_NAMES["Raw"]].iloc[0]
            df[REFL_COLUMN_NAMES["i0Err"]] = average[REFL_COLUMN_NAMES["Raw"]].iloc[0] * df[REFL_COLUMN_NAMES["Stat Update"]]

            df[REFL_COLUMN_NAMES["R"]] = df[REFL_COLUMN_NAMES["Raw"]] / df[REFL_COLUMN_NAMES["i0"]]
            df[REFL_COLUMN_NAMES["R Err"]] = ((izeroDataFrame[REFL_COLUMN_NAMES["Stat Update"]]**2 
                                                           + df[REFL_COLUMN_NAMES["i0Err"]] / df[REFL_COLUMN_NAMES["i0"]]**2) 
                                                           * df[REFL_COLUMN_NAMES["R"]]**2)
        elif kind == "ovp":
            i0, overlaps, done = StitchManager.labelDF(izeroDataFrame, kind, izeroCount, overlaps)
            average = ErrorManager.averageOfAbove(i0, label="overlaps")
            overlaps = pd.concat([average,overlaps], ignore_index=True)
            return i0, overlaps, done
        else:
            raise ValueError("kind must be 'i0' or 'init'")

    @staticmethod
    def stitch(
        currentDataFrame: pd.DataFrame,
        priorDataFrame: pd.DataFrame,
        initialOverlapCount: int,
        overlapPoints: int,
    ):
        '''Scales future stitch dataframes to their prior stitch dataframes'''
        i0, overlaps, done = StitchManager.normalize(currentDataFrame, initialOverlapCount, overlapPoints, kind="ovp")
        priorOverlaps = priorDataFrame.iloc[-overlapPoints:]

        overlaps[REFL_COLUMN_NAMES["i0"]] = overlaps[REFL_COLUMN_NAMES["Raw"]] / priorOverlaps[REFL_COLUMN_NAMES["R"]]

        scale = overlaps[REFL_COLUMN_NAMES["i0"]].mean()
        scaleVar = overlaps[REFL_COLUMN_NAMES["i0"]].var()

        overlaps[REFL_COLUMN_NAMES["R"]] = overlaps[REFL_COLUMN_NAMES["Raw"]] * scale
        overlaps[REFL_COLUMN_NAMES["R Err"]] = overlaps[REFL_COLUMN_NAMES["R"]]**2 * (overlaps[REFL_COLUMN_NAMES["Stat Update"]]**2 + scaleVar / scale**2)

        averageR = (overlaps[REFL_COLUMN_NAMES["R"]] + priorOverlaps[REFL_COLUMN_NAMES["R"]]) / 2
        averageRErr = 1/(overlaps[REFL_COLUMN_NAMES["R Err"]] + priorOverlaps[REFL_COLUMN_NAMES["R Err"]])

        averageDF = pd.DataFrame({REFL_COLUMN_NAMES["R"]: averageR, REFL_COLUMN_NAMES["R Err"]: averageRErr, REFL_COLUMN_NAMES["catagory"]: "done"})

        currentDataFrame = pd.concat([i0, overlaps, averageDF, done], ignore_index=True)


    @staticmethod
    def scaleDataFrame(imageDataFrames, reflDataFrames: list[pd.DataFrame]):
        
        izeroCount, initialOverlapCount, overlaps = OverlapFactory.getOverlap(reflDataFrames[0], reflDataFrames[1])

        for i, df in enumerate(reflDataFrames):
            ErrorManager.updateStats(df, izeroCount)
            df[REFL_COLUMN_NAMES["stitch num"]] = i
            if i == 0:
                StitchManager.normalize(df, izeroCount)
            else:
                StitchManager.stitch(df, reflDataFrames[i - 1], initialOverlapCount, overlaps)
        refl = pd.concat(reflDataFrames, ignore_index=True)
        image = pd.concat(imageDataFrames, ignore_index=True)
        return refl, image