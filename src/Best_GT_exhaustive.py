from typing import Iterable
import numpy as np
from collections import defaultdict
from itertools import product
from Best_IM import Best_IM

class Best_GT_exhaustive():
    # exhaustive
    def __init__(self, m: np.ndarray) -> None:
        self.m = m
        self.len = len(m)
        self.scores = defaultdict(list)
        self.best_g = np.array([]) 
        self.best_g_score = 0
        self.all_best = []

    def find_best(self) -> None:
        """
        Find the best basis for some matrix m.
        Generate all possible invertible gauge transform matrices G and multipy them with m.
        Scores the gauge transforms and bucket sorts them.


        :param m: _description_
        :type m: _type_
        """
        self.__score_bases(self.m)
        self.best_m_score = np.min(list(self.scores.keys()))
        self.all_best = self.scores[self.best_m_score]
        self.best_m = self.all_best[0]


    def __score_bases(self, m: np.ndarray) -> None:
        """
        Generate and score alternative bases.
        Do GM = M' for all g in {g}.

        :param m: square binary matrix
        :type m: _type_
        """
        assert np.all(np.logical_or(m == 1,m == 0)), "values not 0 or 1"
        assert m.shape[0] == m.shape[1], "m is not square"
        assert len(m.shape) == 2, "m is not 2d"

        t = self.__generate_invertible(len(m))
        t2 = np.empty(shape = (len(t), size, size))

        for i, g_i in enumerate(t):
            m2 = np.array(self.__gauge_transform_xor(m, g_i))
            t2[i,:,:] = m2

            score  = self.__scoring_sum(m2)
            self.scores[score].append(m2) # FIXME: Note: Placeholder scoring function.
    
    def __generate_invertible(self, rank: int):
        """
        Generate all invertible matrices
        :param rank: side length of the square matrix
        :return:
        """
        return self.__filter_invertible_matrices(self.__generate_binary_matrices((rank,rank)))


    @staticmethod
    def __scoring_sum(m2: np.ndarray) -> int:
        """
        Placeholder scoring function for a basis.

        :param m2: new basis
        :type m2: np.ndarray
        :return: score, lower = better
        :rtype: int
        """
        return np.sum(m2)
       
    @staticmethod    
    def __gauge_transform_xor(m: np.ndarray, g: np.ndarray):
        """
        Compute the XOR matrix product of two binary matrices.

        Parameters:
        - M: Binary matrix
        - G: Binary matrix

        Returns:
        - XOR matrix product of M and G
        """
        result = np.zeros((m.shape[0], g.shape[1]), dtype=int)

        for i in range(m.shape[0]):
            for j in range(g.shape[1]):
                temp_result = 0
                for k in range(m.shape[1]):
                    temp_result ^= m[i, k] & g[k, j]
                result[i, j] = temp_result

        return result

    @staticmethod
    def __generate_binary_matrices(shape: tuple[int, int]):
        possibilities = list(product([0, 1], repeat=shape[0]*shape[1]))
        matrices = [np.array(matrix).reshape(shape) for matrix in possibilities]
        return matrices
    
    @staticmethod
    def __filter_invertible_matrices(matrices: Iterable[np.ndarray]):
        invertible_matrices = [matrix for matrix in matrices if np.linalg.det(matrix) != 0]
        return invertible_matrices

if __name__ == "__main__":
    seed = 253
    rng = np.random.default_rng(seed)
    n = 3
    s_dataset = rng.integers(2,size=(10,n))
    s_dataset = np.where(s_dataset == 0, -1, 1)
    size = 3

    k = 2
    IM = Best_IM(s_dataset, k)
    IM.find_best_basis()
    print(IM.to_base10())
    M = np.array(IM.basis)
    print(M) 



    B = Best_GT_exhaustive(M)
    B.find_best()
    print()
    print(B.best_m)
    # # print(B.scores)
    # print(B.scores.keys())
    
    # [print(i,"\n") for i in B.all_best]
    # FIXME: these are all the same matrix that can be reached with elementary row and column operations,
    #  maybe instead of checking if something is invertible, we can generate matrices that are only different in the nr of elements
    #  and then filter those before constructing all same ones with elementary row operations
    #  we could then also take the transpose of the initial invertible matrices and do the same

    # FIXME: I wonder if we really need to compute GM to know if M' will be our preferred basis.
    #  Given the matrix multiplication, we can deduce which elements will be operated on and maybe we can know
    #  if there will will be a good separation of spins beforehand