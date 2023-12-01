import numpy as np

def validate_gauge_input(M, G):
    assert np.allclose(M.shape, G.shape), "Shape mismatch"
    assert np.allclose(M.shape, M.T.shape), "Matrix not square"
    assert np.all(np.logical_or(M== 0, M== 1)), "Matrix 1 contains values other than 0 and 1."
    assert np.all(np.logical_or(G== 0, G== 1)), "Matrix 2 contains values other than 0 and 1."



def gauge_transform_product(M, G):
    """
    :param M: Model
    :param G: Gauge transform matrix
    :return:
    """
    validate_gauge_input(M, G)

    return np.fmod(np.dot(M, G), 2)


def gauge_transform_xor_np(M, G):
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
            result[i, j] = np.bitwise_xor.reduce(np.bitwise_xor(M[i, :], G[:, j]))

    return result

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


def xor_matrix_product(M, G):
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
            # xor over operator and then NAND is the same as dot product + modulo 2
            result[i, j] = np.bitwise_xor.reduce(np.bitwise_and(M[i, :], G[:, j]))

    return result