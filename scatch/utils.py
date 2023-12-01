import numpy as np

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