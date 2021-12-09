import unittest
import LinearRegression
import numpy as np


class LinearRegressionTest(unittest.TestCase):

    def test_bad_input_handling(self):
        X1 = [[1, 2], [2, 3]]
        X2 = 3
        y1 = [2 , 3]
        y2 = 4

        with self.assertRaises(Exception):
            LinearRegression.linearRegression(X1)

        with self.assertRaises(Exception):
            LinearRegression.linearRegression(X1, y2)
        
        with self.assertRaises(Exception):
            LinearRegression.linearRegression(X2, y1)

    def test_singularity(self):
        X = [[1, 2, 3],
              [1, 3, 4],
              [1, 4, 5],
              [1, 5, 6],
              [1, 6, 7]]
        y = [0, 1, 0, 2, 2]

        with self.assertRaises(Exception):
            LinearRegression.linearRegression(X,y)

    def test_missing_value_handling(self):
        X1 = [[3, 2, 1],
              [0, 2, 1],
              [3, 2, 0],
              [4, 3, 3],
              [5, 5, 2],
              [np.nan, 3, 2]]
        y1 = [0, 1, 0, 2, 2, 3]

        X2 = [[3, 2, 1],
              [0, 2, 1],
              [3, 2, 0],
              [4, 3, 3],
              [5, 5, 2]]
        y2 = [0, 1, 0, 2, 2]

        betaHeadMatrix1, standardErrors1, confidanceIntervals1 = LinearRegression.linearRegression(X1, y1)
        betaHeadMatrix2, standardErrors2, confidanceIntervals2 = LinearRegression.linearRegression(X2, y2)
        np.testing.assert_array_equal(betaHeadMatrix1, betaHeadMatrix2)
        np.testing.assert_array_equal(standardErrors1, standardErrors2)
        np.testing.assert_array_equal(confidanceIntervals1, confidanceIntervals2)

    if __name__ == '__main__':
        unittest.main()
