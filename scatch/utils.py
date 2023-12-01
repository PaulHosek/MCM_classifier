import numpy as np
from itertools import product

def gauge_transform_xor(M, G):
    """
    Compute the XOR matrix product of two binary matrices.

    Parameters:
    - M: Binary matrix
    - G: Binary matrix

    Returns:
    - XOR matrix product of M and G
    """
    result = np.zeros((M.shape[0], G.shape[1]), dtype=int)

    for i in range(M.shape[0]):
        for j in range(G.shape[1]):
            temp_result = 0
            for k in range(M.shape[1]):
                temp_result ^= M[i, k] & G[k, j]
            result[i, j] = temp_result

    return result


def generate_binary_matrices(shape):
    possibilities = list(product([0, 1], repeat=shape[0]*shape[1]))
    matrices = [np.array(matrix).reshape(shape) for matrix in possibilities]
    return matrices

def filter_invertible_matrices(matrices):
    invertible_matrices = [matrix for matrix in matrices if np.linalg.det(matrix) != 0]
    return invertible_matrices

def generate_invertible(rank):
    """
    Generate all invertible matrices
    :param rank: side length of the square matrix
    :return:
    """
    return filter_invertible_matrices(generate_binary_matrices((rank,rank)))
