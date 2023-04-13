import numpy as np

from xrr_toolkit import scattering_vector
from prsoxs_xrr import Calculate_Integral_Ranges


def data_reduction(Images, Data):
    Intensity_nominal = []
    Q = scattering_vector(Data)
    for image in Images:
        R_x, R_y = Calculate_Integral_Ranges(image)
        Image_nominal = image[R_x[0] : R_x[-1]][R_y[0] : R_y[-1]]
        intensity_nominal = np.sum(Image_nominal)
        Intensity_nominal.append(intensity_nominal)
    Data["Q"] = Q
    Data["Intensity"] = unumpy.uarray(Intensity_nominal, np.sqrt(Intensity_nominal))
    return Data
