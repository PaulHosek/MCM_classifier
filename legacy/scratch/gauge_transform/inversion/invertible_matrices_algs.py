import numpy as np
from itertools import product

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


# # Example: Generate all 2x2 invertible binary matrices
# shape = (2, 2)
# all_matrices = generate_binary_matrices(shape)
# invertible_matrices = filter_invertible_matrices(all_matrices)
#
# for matrix in invertible_matrices:
#     print(matrix)
