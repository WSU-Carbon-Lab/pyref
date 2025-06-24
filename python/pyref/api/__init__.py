"""
Image processing module for Reflectance measurements on CCD Detectors.

This module provides a set of functions for processing images obtained from CCD
detectors, specifically for analyzing reflectance measurements. The goal is to detect
the X-ray beam spot and use it to calculate the specular reflectance from a masked
image.

It also provides pandas extensions for handling image data in DataFrames.
"""

from pyref.api.extensions import ImageAccessor, ImageArray, ImageDtype

__all__ = ["ImageAccessor", "ImageArray", "ImageDtype"]
