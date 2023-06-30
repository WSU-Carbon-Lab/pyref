import sys
import unittest

print(sys.path)
sys.path.append("../xrr")
print(sys.path)
from xrr.Py_xrr import *


# class stitchUnitTest(unittest.TestCase):
#     ...


# if __name__ == "__main__":
#     test_fits = Path().absolute().parent / "TestData" / "TestFits.fits"
#     print(fits.getheader(test_fits, 0))
