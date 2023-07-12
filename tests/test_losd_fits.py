import unittest
import numpy as np
from xrr.load_fits import *

test_fits_file = Path("tests/TestData/TestFits.fits").resolve()
test_multi_path = Path("tests/TestData/Sorted/ZnPc_P100_E180276/282.5/190.0").resolve()

test_df = Path("tests/TestData/TestDataFrame.csv").resolve()
test_single_df = Path("tests/TestData/TestSingleDataFrame.csv").resolve()
test_single_image = Path("tests/TestData/TestSingleImage.txt").resolve()


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

        expected_header_data = pd.read_csv(test_single_df)
        header_data = pd.DataFrame(FitsReader.readHeader(test_fits_file), index=[0])
        self.assertTrue(expected_header_data.equals(header_data))

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
        expected_header_data = expected_header_data.reset_index(drop=True)

        header_data, _ = MultiReader.readFile(test_multi_path)
        header_data = header_data.reset_index(drop=True)

        self.assertTrue(header_data.equals(expected_header_data))

    def test_readHeader(self):
        global test_fits_file
        global test_single_image
        global test_fits_file

        expected_header_data = pd.read_csv(test_df)
        expected_header_data = expected_header_data.reset_index(drop=True)

        header_data = MultiReader.readHeader(test_multi_path)
        header_data = header_data.reset_index(drop=True)

        self.assertTrue(expected_header_data.equals(header_data))


if __name__ == "__main__":
    unittest.main()
