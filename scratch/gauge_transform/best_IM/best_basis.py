import numpy as np

from itertools import product
from collections import defaultdict

##### IM LL ####
def im_ll(b_a, s_dataset):
    """log P(s_hat|g_hat,M)
    log likelihood of IM
    Eq. 18
    """
    ll = 0
    all_m = get_m_bias(b_a, s_dataset)
    print(all_m)
    N = len(all_m)

    for a, m_a in enumerate(all_m):
        ll += h_func(m_a)
    ll *= -1*N
    return ll

def get_m_bias(ba,s_dataset):
    """Returns m_a the bias of the operator b_a applied to s_hat(Eq. 20)."""
    # ba = binary vector with the spins involved
    res = ba * s_dataset
    return np.average(res, axis=1)

def h_func(m):
    """Eq. 19"""
    return -1*(1+m/2)*np.log(1+m/2) - (1-m/2)*np.log(1-m/2)

### Build up IM up to order K ###


def select_most_biased(s_dataset, k):
    n = s_dataset.shape[0]
    k = 2
    all_b_a = generate_binary_combinations(n,k)
    lls = np.array(all_b_a.shape[1]))
    for b_a in all_b_a:
        ll = im_ll(b_a, )




def exclude_combinations():
    pass




def generate_binary_combinations(n, k):
    if k > n:
        raise ValueError("Number of 1s (k) cannot exceed array length (n)")

    combinations = list(product([0, 1], repeat=n))
    valid_combinations = [comb for comb in combinations if sum(comb) <= k and any(comb)]

    return np.array(valid_combinations)



if __name__ == "__main__":
    rng = np.random.default_rng(42)

    row_len = 4
    s_dataset = rng.integers(2,size=(4,4))
    s_dataset = np.where(s_dataset == 0, -1, 1)

    # basis b_a = [1,1,1,0,0...,0]
    b_a = np.zeros(4)
    b_a[[0,1,2]] = 1


    # res = im_ll(b_a, s_dataset)
    # print(res)
    # Example usage:
    n = 4
    k = 2
    result = generate_binary_combinations(n, k)
    print(result)



    




    

    
    
