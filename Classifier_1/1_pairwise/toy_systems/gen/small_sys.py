import numpy as np


seed = 1
def gen_data(spins):

    rng = np.random.default_rng(seed)
    # data = rng.integers(2, size= (10000,8))
    data = rng.integers(0,2,(10000,spins))
    data[:2500,[1,2,3]] = 0

    data[:300, 2] = data[:300,3]
    data[:3000, 1] = data[:3000, 2]

    # data[data==0] = 2
    return data


if __name__ == "__main__":
    spins = 14
    res = gen_data(spins)
#   pij = np.einsum("i,ij,ik->jk")
    np.savetxt(f"{spins}spin.dat", res,fmt="%d", delimiter=" ")
