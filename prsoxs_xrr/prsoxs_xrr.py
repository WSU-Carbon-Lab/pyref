"""Main module."""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

c = 299_792_458 # m s-2
ħ = 6.582_119_569 # eV s

def load_data(Directory):
    """
        Parses every .fits file given in ``files`` and returns the meta and image data
        
        Returns
        -------
        images : list
            List of each image file associated with the .fits
        meta : pd.Dataframe
            pandas dataframe composed of all meta data for each image
        
        """
    files = [f'{Directory}/{filename}' for filename in os.listdir(Directory)]
    temp_meta = {}
    out_images = []
    out_meta = None
    for i, file in enumerate(files):
        with fits.open(file) as hdul:
            header = hdul[0].header
            del header['COMMENT']  # Drop all non-values
            for item in header:
                temp_meta[item] = header[item]
            out_images.append(hdul[2].data)
        if i == 0:
            out_meta = pd.DataFrame(temp_meta, index=[i])
        else:
            out_meta = out_meta.append(pd.DataFrame(temp_meta, index=[i]))

    return out_images, out_meta

def Calculate_Integral_Ranges(Image, edge_trim = (5,5)):
    N_x, N_y = Image.shape
    bright_spot_x, bright_spot_y = np.unravel_index(Image.argmax(), Image.shape)
    temporary_x_range = range(edge_trim[0], N_x - edge_trim[0])
    temporary_y_range = range(edge_trim[1], N_y - edge_trim[1])
    if bright_spot_x in temporary_x_range:
        x_range =  temporary_x_range
    elif bright_spot_x < edge_trim[0]:
        x_range = range(0, N_x - edge_trim[0])
    else:
        x_range = range(edge_trim[0], N_x)
    if bright_spot_y in temporary_y_range:
        y_range = temporary_y_range
    elif bright_spot_y < edge_trim[1]:
        y_range = range(0, N_y - edge_trim[1])
    else:
        y_range = range(edge_trim[1], N_y)
    return x_range, y_range

def scattering_vector(Metadata):
    Energy = Metadata['Beamline Energy']
    Theta = Metadata['Sample Theta']
    k = Energy/ħ/c
    Q = 2*k*np.sin(Theta/2)
    return Q

def metadata_to_scattering(MetaData):
    Important_Information = ['Beamline Energy','Sample Theta',]
    Data = MetaData[Important_Information]
    RealMetaData = MetaData.drop(Important_Information) 
    return Data, RealMetaData
    

def data_reduction(Images, MetaData):
    Intensity = []
    Q = scattering_vector(MetaData)
    for image in Images:
        R_x, R_y = Calculate_Integral_Ranges(image)
        Image = image[R_x[0]:R_x[-1]][R_y[0]:R_y[-1]]
        intensity = np.sum(Image)
        Intensity.append(intensity)
    MetaData['Q'] = Q
    MetaData['Intensity'] = Intensity
    return MetaData

def normalization(Data):
    i_zero_points = pd.DataFrame[Data['Q'] == 0]
    return i_zero_points

if __name__ == '__main__':
    dir = r'C:/Users/hduva/prsoxs_xrr/tests/TestData/Sorted/282.5'
    Images, MetaData = load_data(dir)
    Data, MetaData = metadata_to_scattering(MetaData)
    Data = data_reduction(Data)
    print(normalization(Data))