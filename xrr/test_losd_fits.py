import unittest
import numpy as np
from _load_fits import *

test_fits_file = Path("tests/TestData/TestFits.fits")
test_multi_path = Path("tests/TestData/Sorted/282.5")

test_df = Path("tests/TestData/TestDataFrame.csv")
test_single_df = Path("tests/TestData/TestSingleDataFrame.csv")
test_single_image = Path("tests/TestData/TestSingleImage.txt")


class ReadFitsUnitTest(unittest.TestCase):
    def test_readFile(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_header_data = pd.read_csv(test_single_df)
        expected_image = np.loadtxt(test_single_image)
        header_data, image = FitsReader.readFile(test_fits_file)

        self.assertTrue(header_data.equals(expected_header_data))
        self.assertTrue(np.array_equal(image, expected_image))

    def test_readHeader(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_header_data = pd.read_csv(test_single_df).to_dict()
        header_data = FitsReader.readHeader(test_fits_file)
        self.assertDictEqual(expected_header_data, header_data)

    def test_readImage(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_image = np.loadtxt(test_single_image)
        image = FitsReader.readImage(test_fits_file)
        self.assertTrue(np.array_equal(image, expected_image))


class MultiReaderUnitTest(unittest.TestCase):
    def test_readFile(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_header_data = pd.read_csv(test_df)
        expected_image = np.loadtxt(test_multi_path)
        header_data, image = MultiReader.readFile(test_multi_path)

        self.assertTrue(header_data.equals(expected_header_data))

    def test_readHeader(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_header_data = pd.read_csv(test_df)
        header_data = MultiReader.readHeader(test_multi_path)
        self.assertTrue(expected_header_data.equals(header_data))


if __name__ == "__main__":
    unittest.main()
