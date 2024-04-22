import numpy as np


seed = 1
def gen_data():
    rng = np.random.default_rng(seed)
    # data = rng.integers(2, size= (10000,8))
    data = np.ones((5000,8))
    data[:2500,[1,2,3]] = 0

    data[:300, 2] = data[:300,3]
    data[:3000, 1] = data[:3000, 2]

    data[data==0] = 2
    return data


if __name__ == "__main__":
    res = gen_data()
#   pij = np.einsum("i,ij,ik->jk")
    np.savetxt("8spin_12.dat", res,fmt="%d", delimiter=" ")
