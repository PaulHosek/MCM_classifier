from operator import xor
import numpy as np

from itertools import product, combinations
from collections import defaultdict

##### IM LL ####
def im_ll(ba, s_dataset):
    """log P(s_hat|g_hat,M)
    log likelihood of IM
    Eq. 18
    """
    ll = 0
    all_m = get_m_bias(ba, s_dataset)

    N = len(all_m)

    for a, m_a in enumerate(all_m):
        ll += h_func(m_a)
    ll *= -1*N
    return ll

def get_m_bias(ba,s_dataset):
    """Returns m_a the bias of the operator ba applied to s_hat(Eq. 20)."""
    # ba = binary vector with the spins involved
    res = ba * s_dataset
    return np.average(res, axis=1)

def h_func(m):
    """Eq. 19"""
    return -1*(1+m/2)*np.log(1+m/2) - (1-m/2)*np.log(1-m/2)

### Build up IM up to order K ###
def find_best_basis(s_dataset,k):
    basis = list()
    n = s_dataset.shape[0]
    print(s_dataset)
    ba_combs = generate_binary_combinations(n,k)




    exclude_mask = np.ones(ba_combs.shape[0],dtype=bool)

    # for the first 2 there 
    for i in range(2):
        best_ba, best_idx, _ = select_most_biased_ba(s_dataset, k, ba_combs)
        basis.append(best_ba)
        exclude_mask[best_idx] = False


    # update exclude mask with the rows/binary vectors that are in the new combinations of the selected basis elements    
    exclude_mask = exclude_mask & ~(ba_combs[:, None] == excluded_combinations(basis)).all(-1).any(-1) 


    best_ba, best_idx, _ = select_most_biased_ba(s_dataset, k, ba_combs)
    basis.append(best_ba)
    # improvement: do not need to check the ones that are already in the exclude mask, would be smarter to just change the exclude mask directly with the all_ba
    exclude_mask[best_idx] = False
    exc = excluded_combinations(basis)




# !! Cannot use this because the index does not match the number
# def arr_to_int(binary_arr):
#     return binary_arr.dot(1 << np.arange(binary_arr.shape[-1] - 1, -1, -1))


def select_most_biased_ba(s_dataset, k, val_combs):
    # find 
    n = s_dataset.shape[0]
    k = 2

    best_ba = np.empty(val_combs.shape[1])
    best_ll = np.NINF
    best_idx = 0
    for i, ba in enumerate(val_combs):
        cur = im_ll(ba,s_dataset)
        if cur > best_ll:
            best_ll = cur
            best_ba = ba
            best_idx = i
    return best_ba, best_idx, best_ll



def excluded_combinations(bas):
    # between every 2 vectors in binary vectors, do element wise xor operation.
    # goal: get all possible interactions that could be build from these and return the exclude list
    all_comb = combinations(bas, 2)
    exclude = list()
    for comb in all_comb:
        exclude.append(np.logical_xor(comb[0],comb[1]).astype(int))
    
    return np.array(exclude)



def generate_binary_combinations(n, k):
    # build up all interaction vectors of lengh n and interaction order k
    if k > n:
        raise ValueError("Number of 1s (k) cannot exceed array length (n)")

    combinations = list(product([0, 1], repeat=n))
    valid_combinations = [comb for comb in combinations if sum(comb) <= k and any(comb)]

    return np.array(valid_combinations)



if __name__ == "__main__":

    # ba_combs = np.array([[0, 0, 0, 1],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 1, 1],
    #                     [0, 1, 0, 0],
    #                     [0, 1, 0, 1],
    #                     [0, 1, 1, 0],
    #                     [1, 0, 0, 0],
    #                     [1, 0, 0, 1],
    #                     [1, 0, 1, 0],
    #                     [1, 1, 0, 0]])
    
    # exc = np.array([[0, 0, 0, 1],
    #                 [0, 0, 1, 0]]) 

    # res = ~(ba_combs[:, None] == exc).all(-1).any(-1)

    # exclude = ~np.array([False]*9 + [True]*1)
    # mask = res & exclude

    rng = np.random.default_rng(42)

    row_len = 4
    s_dataset = rng.integers(2,size=(4,4))
    s_dataset = np.where(s_dataset == 0, -1, 1)


    # basis ba = [1,1,1,0,0...,0]
    ba = np.zeros(4)
    ba[[0,1,2]] = 1
    k = 2
    n = 4


    # print(select_most_biased(s_dataset, k))
    # res = im_ll(ba, s_dataset)
    # print(res)
    # # Example usage:
    # n = 4
    # k = 2
    # result = generate_binary_combinations(n, k)
    # # print(result)
    # res = exclude_combinations(result)
    # Example usage:
    # binary_vectors = np.array([[0, 0, 1, 1], [0, 1, 1, 0],[1,0,0,0]])

    # result = exclude_combinations(binary_vectors)
    # print(result)
    find_best_basis(s_dataset,2)




    




    

    
    
