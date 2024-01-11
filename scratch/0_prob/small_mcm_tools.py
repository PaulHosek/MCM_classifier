import matplotlib.pyplot as plt
import array
from itertools import product
from re import I
import numpy as np

def generate_class_dataset(cls):
    """Generate fake data for the T class for  a square grid of size n
    - classes equivalent in nr of modeled spins and nr 0s 
    Top-class
    [a,b,c]
    [d,e,f]
    [0,0,0]

    Corner_class
    [0,a,b]
    [c,0,d]
    [e,f,0]

    :param n: sidelength. 3 or 4
    :param cls: class. t or c
    """


    if cls.lower() == "t":

        upper = __generate_binary_matrices((2,3))
        matrices = [np.vstack((i,np.array([0,0,0]))) for i in upper]
        return matrices
    
    elif cls.lower() == "c":
        upper = __generate_binary_matrices((2,3))
        matrices = [np.vstack((i,np.array([0,0,0]))) for i in upper]
        return [top_to_corner(m) for m in matrices]
        # transformed_arrays = [np.eye(n, dtype=int)[:-1, :] arr for arr in matrices]
    else:
        raise ValueError("Class must be either t or c")


def top_to_corner(top_matrix):
    corner_matrix = np.zeros(top_matrix.shape)
    corner_matrix[0, 1] = top_matrix[0, 0]
    corner_matrix[0, 2] = top_matrix[0, 1]

    corner_matrix[1, 0] = top_matrix[0, 2]
    corner_matrix[1, 2] = top_matrix[1, 0]

    corner_matrix[2, 0] = top_matrix[1, 1]
    corner_matrix[2, 1] = top_matrix[1, 2]

    return corner_matrix


def __generate_binary_matrices(shape: tuple[int, int]):
    possibilities = list(product([0, 1], repeat=shape[0]*shape[1]))
    matrices = [np.array(matrix).reshape(shape) for matrix in possibilities]
    return matrices

def arr_to_int(binary_arr):
    return int(binary_arr.dot(1 << np.arange(binary_arr.shape[-1] - 1, -1, -1)))
