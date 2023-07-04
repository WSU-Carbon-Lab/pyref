import unittest
import numpy as np
from _load_fits import *

test_fits_file = Path("tests/TestData/TestFits.fits")
test_multi_path = Path("tests/TestData/Sorted/282.5")

test_df = Path("tests/TestData/TestDataFrame.csv")
test_single_df = Path("tests/TestData/TestSingleDataFrame.csv")
test_single_image = Path("tests/TestData/TestSingleImage.txt")


class LoadFitsUnitTest(unittest.TestCase):
    def test_loadFits(self):
        global test_fits_file

        expected_header_data = pd.read_csv(test_single_df)
        expected_image = np.loadtxt(test_single_image)
        header_data, image = loadFits(test_fits_file)

        self.assertTrue(header_data.equals(expected_header_data))
        self.assertTrue(np.array_equal(image, expected_image))

    def test_loadFits_flags(self):
        global test_fits_file
        expected_header_data = pd.DataFrame({}, index=[0])
        expected_image = []

        header_data, image = loadFits(test_fits_file, image=False, header=False)

        self.assertTrue(header_data.equals(expected_header_data))
        self.assertEqual(expected_image, image)


class LoadMultipleFitsUnitTest(unittest.TestCase):
    def test_loadMultipleFits(self):
        global test_df

        expected_header_data = pd.read_csv(test_df)

        header_data, _ = loadMultipleFits(test_multi_path)

        self.assertTrue(header_data.equals(expected_header_data))


if __name__ == "__main__":
    unittest.main()
