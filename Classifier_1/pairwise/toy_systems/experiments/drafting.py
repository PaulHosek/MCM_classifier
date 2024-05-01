import numpy as np
from numba import jit

def calc_energy(fields,coup, state): # state assumed to be in convention -1 1
    h = __calc_fields(fields, state)
    j = __calc_couplings(coup,state)
    return -1*(h+j)


@jit("(float64[:], int64[:])", nopython=True) # state assumed to be in convention -1 1
def __calc_fields(fields, state) -> np.double:
    return np.sum(fields * state)    # FIXME check if this is correct based on the convention

def __calc_couplings(coup, state)->np.double:
    """Compute -1*sum(J_ij*s[i]*s[j]) for all (i,j) with i < j.
    Outer product and masking triangle.
    """
    return np.sum(coup*((state[:,None]*state)[~np.tri(len(state),dtype=bool)]))


def partiton_function():
    # 1. Calculate energy for every observed state
    
    pass



if __name__ == "__main__":
    state = np.array([-1,1,1,-1,-1])
    rng = np.random.default_rng(seed=1)
    N = len(state)
    fields = rng.random((N),dtype=np.double)
    coup =  rng.random(int(N*(N-1)/2),dtype=np.double)    
    print(fields,coup)
    res = calc_energy(fields, coup, state)
    print(res)
