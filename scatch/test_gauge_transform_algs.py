import numpy as np
import unittest
from gauge_transform_algs import *


class TestXORMatrixProduct(unittest.TestCase):

    def test_identity_matrix(self):
        M = np.array([[1, 0], [0, 1]])
        G = np.identity(2, dtype=int)
        result = xor_matrix_product(M, G)
        expected_result = gauge_transform_product(M, G)
        np.testing.assert_array_equal(result, expected_result)

    def test_zero_matrix(self):
        M = np.array([[1, 0], [0, 1]], dtype=int)
        G = np.zeros((2, 2), dtype=int)
        result = xor_matrix_product(M, G)
        expected_result = gauge_transform_product(M, G)
        np.testing.assert_array_equal(result, expected_result)

    def test_random_matrices(self):
        M = np.random.randint(2, size=(3, 3)).astype(bool).astype(int)
        G = np.random.randint(2, size=(3, 3)).astype(bool).astype(int)
        result = xor_matrix_product(M, G)
        expected_result = gauge_transform_product(M, G)
        np.testing.assert_array_equal(result, expected_result)

    def test_large_matrices(self):
        M = np.random.randint(2, size=(100, 100)).astype(bool)
        G = np.random.randint(2, size=(100, 100))
        result = xor_matrix_product(M, G)
        expected_result = gauge_transform_product(M, G)
        np.testing.assert_array_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
