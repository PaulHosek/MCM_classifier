import numpy as np
from itertools import product, combinations, chain
import timeit
import matplotlib.pyplot as plt

class Best_IM():
    def __init__(self, s_dataset, k) -> None:
        self.s_dataset = s_dataset
        self.k = k
        self.n = s_dataset.shape[1]
        self.basis = list()   


    def find_best_basis(self) -> None:
        """Find the best basis with interactions up to order k for the give data set.
        1. Build up all possible basis operators ba.
        2. While rank < n: select next most biased basis operator
        3. Put ba and aall linear combinations with the operators in the basis
        and the higher order operators already excluded into the list to exclude.
        4. Filter the new set of valid ba canidates based on the excluded oeprators

        Note: Test the validity of this function with the is valid_basis function

        :param s_dataset: 2d np array with s_bar as rows
        :type s_dataset: np.ndarray
        :param k: up to which order of interactions we want to consider
        :type k: int
        :sets self.basis: basis: 2d np array with most biased set of independent basis operators ba
        :rtype: np.ndarray
        """

        exclude_combs = list()
        
        ba_combs = self.__generate_binary_combinations(self.n,self.k)
        exclude_mask = np.ones(ba_combs.shape[0],dtype=bool)
        ba_combs_idx= np.arange(ba_combs.shape[0])

        for r in range(1, self.n+1):
            valid_combs = ba_combs[exclude_mask]
            best_ba, best_idx, ll = self.select_most_biased_ba(valid_combs)
            valid_idx = ba_combs_idx[exclude_mask]
            exclude_mask[valid_idx[best_idx]] = False

            # we only need to compute the combinations to exclude based on the new element
            #  with the basis and with the excluded elements

            # exclude higher order interactions
            if len(exclude_combs): 
                new_x_exclude = self.gen_pairwise_interactions(best_ba,exclude_combs)
                exc_idx1 = np.array(np.where((ba_combs[:, None] ==new_x_exclude).all(-1).any(-1)))
                exclude_combs.extend(new_x_exclude)
                exclude_mask[exc_idx1] = False 

            # exclude pairwise interactions
            if len(self.basis):
                new_x_basis = self.gen_pairwise_interactions(best_ba, self.basis)
                exc_idx2 = np.array(np.where((ba_combs[:, None] ==new_x_basis).all(-1).any(-1)))
                exclude_combs.extend(new_x_basis)
                exclude_mask[exc_idx2] = False

            exclude_combs.append(best_ba)
            self.basis.append(best_ba)




    ### BASIS SELECTION TOOLS ###
    def gen_pairwise_interactions(self, ba,ba_collection):
        """Generate all pairwise combinations between ba and all elements in list_ba
        ba = 1d np array
        list_ba = 2d np array of bas
        """

        exclude = [np.logical_xor(ba, i_ba).astype(int) for i_ba in ba_collection]
        return np.unique(exclude, axis=0)

    def select_most_biased_ba(self, val_combs):
        """Select next most biased basis operator."""
        n = self.s_dataset.shape[0]

        best_ba = np.empty(val_combs.shape[1])
        best_ll = np.NINF
        best_idx = 0
        for i, ba in enumerate(val_combs):
            cur = self.__im_ll(ba)
            if cur > best_ll:
                best_ll = cur
                best_ba = ba
                best_idx = i

        return best_ba, best_idx, best_ll

    @staticmethod
    def __generate_binary_combinations(n, k):
        """Generate all binary vectors ba of length n and interaction order ≤k"""

        if k > n:
            raise ValueError("Number of 1s (k) cannot exceed array length (n)")

        combinations = list(product([0, 1], repeat=n))
        valid_combinations = [comb for comb in combinations if sum(comb) <= k and any(comb)]

        return np.array(valid_combinations)
    
    def to_base10(self):
        """Convert 2d binary basis into an integer array"""
        basis = np.array(self.basis)
        return basis.dot(1 << np.arange(basis.shape[-1] - 1, -1, -1))



    ##### IM LL ####
    
    def __im_ll(self, ba):
        """log P(s_hat|g_hat,M)
        log likelihood of IM
        Eq. 18
        """
        ll = 0
        all_m = self.__get_m_bias(ba)
        N = len(all_m)

        for _, m_a in enumerate(all_m):
            ll += self.__h_func(m_a)
        ll *= -1*N
        return ll

    def __get_m_bias(self, ba):
        """Returns m_a the bias of the operator ba applied to s_hat(Eq. 20)."""
        # ba = binary vector with the spins involved
        res = ba * self.s_dataset
        return np.average(res, axis=1)

    @staticmethod
    def __h_func(m):
        """Eq. 19"""
        return -1*(1+m/2)*np.log(1+m/2) - (1-m/2)*np.log(1-m/2)

    ### TESTING ### 
    def is_valid_basis(self):
        """Test if generated basis is valid, by testing all ba in the bassis are
        not a linear combination of any of the other ba in that basis. All interactions are considered."""
        # build up combinations and apply linear combination operator xor to each
        combs = list(self.__powerset_higherorder(self.basis))
        reduced_combs = np.array([tuple(np.logical_xor.reduce(arrays).astype(int)) for arrays in combs])
        # find if any of these operators are in the original basis

        row_mask = (reduced_combs[:, None] == self.basis).all(-1).any(-1)


        if row_mask.any():
            print("invalid ba found.")
            print("combs[res]:", reduced_combs[row_mask])
            print("row mask",row_mask)
            whr = np.where(row_mask == True)

        return not row_mask.any()

    @staticmethod
    def __powerset_higherorder(iterable):
        """
        Gives powerset but excluding empty set and single element sets.
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        powerset_higherorder([1,2,3]) --> (1,2) (1,3) (2,3) (1,2,3)
        
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))



if __name__ == "__main__":
    seed = 253
    rng = np.random.default_rng(seed)
    n = 5
    s_dataset = rng.integers(2,size=(10,n))
    s_dataset = np.where(s_dataset == 0, -1, 1)

    k = 2
    B = Best_IM(s_dataset, k)
    B.find_best_basis()
    
    print("best_basis",B.basis)
    print("Is this basis only made up of independent operators?",B.is_valid_basis())

    # def run_experiment(n):
    #     rng = np.random.default_rng()
    #     s_dataset = rng.integers(2, size=(10, n))
    #     s_dataset = np.where(s_dataset == 0, -1, 1)

    #     k = 2
    #     B = Best_IM(s_dataset, k)

    #     time_taken = timeit.timeit(lambda: B.find_best_basis(), number=1)

    #     return time_taken

    # n_values = np.arange(20)[2:]
    # runtimes = []

    # for n in n_values:
    #     runtime = run_experiment(n)
    #     runtimes.append(runtime)

    # # Plot the results
    # plt.figure
    # plt.plot(n_values, runtimes, marker='o')
    # plt.title('Runtime vs. n')
    # plt.xlabel('n')
    # plt.ylabel('Runtime (seconds)')
    # plt.show()

    







### LEGACY but may come in handy later ###

# !! Cannot use this because the index does not match the number
# def arr_to_int(binary_arr):
#     return binary_arr.dot(1 << np.arange(binary_arr.shape[-1] - 1, -1, -1))

# def independence_valid(basis):
#     """Test if the found basis has only independent operators."""
#     # comb = excluded_combinations(basis) # TODO not enough
#     # overlap = ~(basis[:, None] ==comb).all(-1).any(-1)
#     pass