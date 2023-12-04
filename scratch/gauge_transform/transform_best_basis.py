import numpy as np
from collections import defaultdict

from transform.gauge_transform_algs import gauge_transform_xor
from inversion.invertible_matrices_algs import generate_invertible



# build all possible gauge transforms for this size
# apply all gauge transforms
# find the one that is


def reconstruct_gauge_tranform(m1,m2):
    pass


if __name__ == "__main__":
    size = 3
    # m = np.random.randint(0,2,(3,3),dtype=int)
    m = np.identity(size,dtype=int)
    print(m)
    t = generate_invertible(len(m))
    t2 = np.empty(shape=(len(t),size,size))
    scores = defaultdict(list)
    for i, g_i in enumerate(t):
        m2 = np.array(gauge_transform_xor(m,g_i))
        t2[i,:,:] = m2
        print(m2, "\n")
        scores[np.sum(m2)].append(m2)


        # score M'
        # score_m2 = np.sum(m2)
    # print(np.sum(np.array(t2),axis=0))

    # apply all gauge transforms
    # M: row= spin variable, column = IM/spin operator
    print(scores.keys())
    print("3")
    [print(i,"\n") for i in scores[3]]
    # FIXME: these are all the same matrix that can be reached with elementary row and column operations,
    #  maybe instead of checking if something is invertible, we can generate matrices that are only different in the nr of elements
    #  and then filter those before constructing all same ones with elementary row operations
    #  we could then also take the transpose of the initial invertible matrices and do the same

    # FIXME: I wonder if we really need to compute GM to know if M' will be our preferred basis.
    #  Given the matrix multiplication, we can deduce which elements will be operated on and maybe we can know
    #  if there will will be a good separation of spins beforehand