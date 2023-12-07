from cgitb import reset
import numpy as np
from itertools import product, combinations, chain


### Build up IM up to order K ###
def find_best_basis(s_dataset,k):
    basis = list()

    n = s_dataset.shape[1] # nr of independent operators that should build a basis, equivalent to the size of s_bar.
    ba_combs = generate_binary_combinations(n,k)
    exclude_mask = np.ones(ba_combs.shape[0],dtype=bool)
    print(ba_combs.shape)
    ba_combs_idx= np.arange(ba_combs.shape[0])
    exclude_combs = list()

    for r in range(1, n+1):
        valid_combs = ba_combs[exclude_mask]

        best_ba, best_idx, ll = select_most_biased_ba(s_dataset, valid_combs)


        valid_idx = ba_combs_idx[exclude_mask]
        exclude_mask[valid_idx[best_idx]] = False

        # we only need to compute the combinations to exclude based on the new element
        #  with the basis and with the excluded elements
        if len(exclude_combs): # TODO: problem is right now that we dont get 3 way interactions or higher possibly.
            # exclude higher order interactions
            new_x_exclude = gen_pairwise_interactions(best_ba,exclude_combs) # this works as intended
            exc_idx1 = np.array(np.where((ba_combs[:, None] ==new_x_exclude).all(-1).any(-1)))
            # print((ba_combs[:, None] ==new_x_exclude).all(-1).any(-1))
            # print(exc_idx1)
            

            exclude_combs.extend(new_x_exclude)
            exclude_mask[exc_idx1] = False # (ba_combs[:, None] ==new_x_exclude).all(-1).any(-1)

        if len(basis):
            # exclude pairwise interactions
            new_x_basis = gen_pairwise_interactions(best_ba, basis)
            exc_idx2 = np.array(np.where((ba_combs[:, None] ==new_x_basis).all(-1).any(-1)))
            exclude_combs.extend(new_x_basis)
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
    """Select next most biased basis operator."""
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
    """Generate all binary vectors ba of length n and interaction order ≤k"""

    if k > n:
        raise ValueError("Number of 1s (k) cannot exceed array length (n)")

    combinations = list(product([0, 1], repeat=n))
    valid_combinations = [comb for comb in combinations if sum(comb) <= k and any(comb)]

    return np.array(valid_combinations)


### TESTING ### 

def is_valid_basis(basis):
    """Test if generated basis is valid, by testing all ba in the bassis are
       not a linear combination of any of the other ba in that basis. All interactions are considered."""
    # build up combinations and apply linear combination operator xor to each
    combs = list(powerset_higherorder(basis))
    reduced_combs = np.array([tuple(np.logical_xor.reduce(arrays).astype(int)) for arrays in combs])
    # find if any of these operators are in the original basis

    row_mask = (reduced_combs[:, None] == basis).all(-1).any(-1)


    if row_mask.any():
        print("invalid ba found.")
        print("combs[res]:", reduced_combs[row_mask])
        print("row mask",row_mask)
        whr = np.where(row_mask == True)

    return not row_mask.any()


    # return not any_non_indp

def powerset_higherorder(iterable):
    """
    Gives powerset but excluding empty set and single element sets.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    powerset_higherorder([1,2,3]) --> (1,2) (1,3) (2,3) (1,2,3)
    
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))



if __name__ == "__main__":


    for i in range(100):
        rng = np.random.default_rng(0)

        s_dataset = rng.integers(2,size=(10,11))
        s_dataset = np.where(s_dataset == 0, -1, 1)

        k = 2
        n = 4

        best_basis = find_best_basis(s_dataset,4)
        print("best_basis",best_basis)

                # print(len(best_basis))

        print(is_valid_basis(np.array(best_basis)))
        if not is_valid_basis(np.array(best_basis)):
            print(i)
            break 
    


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

    # array1 = np.array([[1, 2, 3],
    #                 [7, 8, 9]])

    # array2 = np.array([[4, 5, 6],
    #                 [10, 11, 12],
    #                 [13, 14, 15]])

    # Check if any rows from array1 are in array2
    # rows_in_array2 = np.isin(array1, array2).all(axis=1)

    # If any row in array1 is in array2, rows_in_array2 will be True
    # print("Rows from array1 present in array2:", rows_in_array2.any())

    # combs = list(powerset(binary_vectors))[1:]
    # result = [tuple(np.logical_xor.reduce(arrays).astype(int)) for arrays in combs]

    # # Example output
    # for res in result:
    #     print(res)
    # new_x_basis = gen_pairwise_interactions(np.array([1,1,1,1]), binary_vectors)
    # mask1 = np.where((ba_combs==new_x_basis[:,None]).all(-1))[1]

    # for i in range(exc.shape[0]):
    #     print(exc[i,:])

    # tes = independence_valid(binary_vectors)
    # print(tes)

    # result = exclude_combinations(binary_vectors)
    # print(result)

    











### LEGACY but may come in handy later ###

# !! Cannot use this because the index does not match the number
# def arr_to_int(binary_arr):
#     return binary_arr.dot(1 << np.arange(binary_arr.shape[-1] - 1, -1, -1))

# def independence_valid(basis):
#     """Test if the found basis has only independent operators."""
#     # comb = excluded_combinations(basis) # TODO not enough
#     # overlap = ~(basis[:, None] ==comb).all(-1).any(-1)
#     pass