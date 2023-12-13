from itertools import product
from re import I
import numpy as np

# goal here is to construct 2 toy classes on a very small grid where we can do exhaustive search for the MCM
# we set some pixels to 0 or set a configuration that is not observed in the test set to 0
# we want to know what happens what we can still say about the shape we observe if we e.g., cut out a pixel that is 0 in the test data


def generate_class_dataset(n,cls):
    """Generate fake data for the T class for  a square grid of size n
    - classes equivalent in nr of modeled spins and nr 0s 
    Top-class
    [a,b,e]
    [c,d,f]
    [0,0,0]

    Corner_class
    [0,a,b]
    [c,0,e]
    [f,g,0]

    :param n: sidelength. 3 or 4
    :param cls: class. t or c
    """

    assert n in [3,4], "n must be 3 or 4"
    assert cls.lower() in ["t","c"]

    if n == 3:
        if cls.lower() == "t":

            upper = __generate_binary_matrices((2,3))
            matrices = [np.vstack((i,np.array([0,0,0]))) for i in res]
            return matrices
        
        elif cls.lower() == "c":
            upper = __generate_binary_matrices((2,3))
            matrices = [np.vstack((i,np.array([0,0,0]))) for i in res]
            # transformed_arrays = [np.eye(n, dtype=int)[:-1, :] arr for arr in matrices]





def __generate_binary_matrices(shape: tuple[int, int]):
    possibilities = list(product([0, 1], repeat=shape[0]*shape[1]))
    matrices = [np.array(matrix).reshape(shape) for matrix in possibilities]
    return matrices


def bottom_to_diag():
    pass

if __name__ == "__main__":
    res = __generate_binary_matrices((2,3))
    matrices = [np.vstack((i,np.array([0,0,0]))) for i in res]
    # transformed_arrays = [np.eye(arr.shape[0], dtype=int)[:-1, :] for arr in matrices]
    # transformed_arrays = np.eye(3,dtype=int)
    m = np.array(matrices)
    # f2 = m[-2,:,:][None, ...]
    f2 = m[-2:,:,:]
    print(f2)
    print()
    print(f2.swapaxes(0,1))
    

    # res = np.vstack(res[],np.array([0,0,0],axis=2))
    # print(res)

            

        
