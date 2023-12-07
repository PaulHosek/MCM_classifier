from operator import xor
import numpy as np

from itertools import product, combinations
from collections import defaultdict

### Build up IM up to order K ###
def find_best_basis(s_dataset,k):
    basis = list()

    n = s_dataset.shape[1] # nr of independent operators that should build a basis, equivalent to the size of s_bar.
    ba_combs = generate_binary_combinations(n,k)
    exclude_mask = np.ones(ba_combs.shape[0],dtype=bool)
    ba_combs_idx= np.arange(len(ba_combs))
    exclude_combs = list()

    for r in range(1, n+1):
        valid_combs = ba_combs[exclude_mask]

        best_ba, best_idx, ll = select_most_biased_ba(s_dataset, valid_combs)


        valid_idx = ba_combs_idx[exclude_mask]
        exclude_mask[valid_idx[best_idx]] = False

        # print(exclude_mask.astype(int), best_idx)
        # after there is 1 element in the basis list and we computed the next, exclude combinations
        if len(basis):
            if len(exclude_combs):
                # exclude higher order interactions
                new_x_exclude = gen_pairwise_interactions(best_ba,exclude_combs)
                exc_idx1 = np.where((ba_combs==new_x_exclude[:,None]).all(-1))[1]
                exclude_mask[exc_idx1] = False

            # exclude pairwise interactions
            new_x_basis = gen_pairwise_interactions(best_ba, basis)
            exc_idx2 = np.where((ba_combs==new_x_basis[:,None]).all(-1))[1]
            exclude_mask[exc_idx2] = False


        exclude_combs.append(best_ba)
        basis.append(best_ba)

    return basis

def gen_pairwise_interactions(ba,ba_collection):
    """Generate all pairwise combinations between ba and all elements in list_ba
    ba = 1d np array
    list_ba = 2d np array of bas
    """

    exclude = [np.logical_xor(ba, i_ba).astype(int) for i_ba in ba_collection]
    return np.unique(exclude, axis=0)



def independence_valid(basis):
    """Test if the found basis has only independent operators."""
    # comb = excluded_combinations(basis) # TODO not enough
    # overlap = ~(basis[:, None] ==comb).all(-1).any(-1)
    pass

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


### basis selection ###

def select_most_biased_ba(s_dataset, val_combs):
    n = s_dataset.shape[0]

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




def generate_binary_combinations(n, k):
    # build up all interaction vectors of lengh n and interaction order k
    if k > n:
        raise ValueError("Number of 1s (k) cannot exceed array length (n)")

    combinations = list(product([0, 1], repeat=n))
    valid_combinations = [comb for comb in combinations if sum(comb) <= k and any(comb)]

    return np.array(valid_combinations)


# !! Cannot use this because the index does not match the number
# def arr_to_int(binary_arr):
#     return binary_arr.dot(1 << np.arange(binary_arr.shape[-1] - 1, -1, -1))
def is_valid_basis(matrix):
    """Test if generated basis is valid, by testing if it is not a linear combination."""
    rows, cols = matrix.shape
    matrix_mod = matrix.copy()

    for i in range(rows):
        # Check if the current row is a linear combination of previous rows
        for j in range(i):
            # Find a non-zero element in the current row
            nonzero_indices = np.nonzero(matrix_mod[i, :])[0]
            if len(nonzero_indices) > 0:
                # Calculate the scaling factor
                scaling_factor = matrix_mod[i, nonzero_indices[0]] / matrix_mod[j, nonzero_indices[0]]
                
                # Update the current row using the linear combination
                if scaling_factor != 0:
                    matrix_mod[i, :] = np.logical_xor(matrix_mod[i, :], scaling_factor * matrix_mod[j, :])

    # Check if any row is all zeros, meaning it's a linear combination of previous rows
    is_combination = np.any(np.all(matrix_mod == 0, axis=1))
    
    return not is_combination



if __name__ == "__main__":


    # for i in range(50):
    rng = np.random.default_rng(32)

    s_dataset = rng.integers(2,size=(10,10))
    s_dataset = np.where(s_dataset == 0, -1, 1)

    k = 2
    n = 4

    best_basis = find_best_basis(s_dataset,3)
    print(best_basis)
    print(len(best_basis))

    print(is_valid_basis(np.array(best_basis)))
    # if not is_valid_basis(np.array(best_basis)):
        # print(i)
        # break




    # print(select_most_biased(s_dataset, k))
    # res = im_ll(ba, s_dataset)
    # print(res)
    # # Example usage:
    # n = 4
    # k = 2
    # result = generate_binary_combinations(n, k)
    # # print(result)
    # res = (result)
    # Example usage:
    # exc = np.array([[0,0,1,1]])
    # ba_combs = generate_binary_combinations(4,2)
    # binary_vectors = np.array([[0, 1, 1, 0],[1,0,0,0],[0,0,0,1]])

    # new_x_basis = gen_pairwise_interactions(np.array([1,1,1,1]), binary_vectors)
    # mask1 = np.where((ba_combs==new_x_basis[:,None]).all(-1))[1]

    # for i in range(exc.shape[0]):
    #     print(exc[i,:])

    # tes = independence_valid(binary_vectors)
    # print(tes)

    # result = exclude_combinations(binary_vectors)
    # print(result)

    













