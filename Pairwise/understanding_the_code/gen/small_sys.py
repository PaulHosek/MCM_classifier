import numpy as np


def gen_2x2():
    data = np.random.randint(2, size= (121,121))

    data[:30, 2] = data[:30,3]
    return data


if __name__ == "__main__":
    res = gen_2x2()
#   pij = np.einsum("i,ij,ik->jk")
    np.savetxt("121spin.dat", res,fmt="%d", delimiter=" ")
