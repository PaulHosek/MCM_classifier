import unittest
import numpy as np
from invertible_matrices_algs import *

class TestInvertibleMatrices(unittest.TestCase):

    def test_generate_binary_matrices(self):
        shape = (2, 2)
        matrices = generate_binary_matrices(shape)

        # Check if the number of generated matrices is correct
        self.assertEqual(len(matrices), 2**(shape[0]*shape[1]))

        # Check if all matrices have the correct shape
        for matrix in matrices:
            self.assertEqual(matrix.shape, shape)

    def test_filter_invertible_matrices(self):
        matrices = [
            np.array([[1, 0], [0, 1]]),  # Invertible
            np.array([[1, 1], [0, 1]]),  # Invertible
            np.array([[0, 0], [0, 0]]),  # Not invertible (determinant is 0)
            np.array([[1, 1], [1, 1]])   # Not invertible (determinant is 0)
        ]

        invertible_matrices = filter_invertible_matrices(matrices)

        # Check if only invertible matrices are retained
        self.assertEqual(len(invertible_matrices), 2)

        # Check if all matrices in the result are indeed invertible
        for matrix in invertible_matrices:
            self.assertNotEqual(np.linalg.det(matrix), 0)

if __name__ == '__main__':
    unittest.main()
