"""Main module."""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def load_data(directory) -> tuple:
    files = f'{directory}/{os.listdir(directory)}'
    Images = []
    MetaData = []
    for file in files:
        with fits.open(file) as f:
            Header = f[0].header
            temp_metadata =[meta for meta in Header]
            del Header['COMMENT']
            
            Images.append(f[2].data)
        MetaData.append(temp_metadata)
    return  Images, MetaData

if __name__ == '__main__':
    dir = r'C:/Users/hduva/prsoxs_xrr/tests/TestData/Sorted/282.5'
    Images, MetaData = load_data(dir)
    plt.imshow(Images[0])
    