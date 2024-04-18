import numpy as np


seed = 1
def gen_data():
    rng = np.random.default_rng(seed)
    data = rng.integers(2, size= (1000,4))

    data[:30, 2] = data[:30,3]
    data[data==0] = -1
    return data


if __name__ == "__main__":
    res = gen_data()
#   pij = np.einsum("i,ij,ik->jk")
    np.savetxt("4spin_conv-11.dat", res,fmt="%d", delimiter=" ")
