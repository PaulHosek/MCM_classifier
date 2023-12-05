import numpy as np
from collections import defaultdict

from itertools import product
from transform.gauge_transform_algs import gauge_transform_xor
from inversion.invertible_matrices_algs import generate_invertible


class Best_basis():
    def __init__(self, m) -> None:
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
        self.__score_bases(m)
        self.best_m_score = np.min(list(self.scores.keys()))
        self.all_best = self.scores[self.best_g_score]
        self.best_m = self.all_best[0]


    def __score_bases(self, m) -> None:
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

    @staticmethod
    def __scoring_sum(m2) -> float:
        return np.sum(m2)
       
    @staticmethod    
    def __gauge_transform_xor(m, g):
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
    def __generate_binary_matrices(shape):
        possibilities = list(product([0, 1], repeat=shape[0]*shape[1]))
        matrices = [np.array(matrix).reshape(shape) for matrix in possibilities]
        return matrices
    
    @staticmethod
    def __filter_invertible_matrices(matrices):
        invertible_matrices = [matrix for matrix in matrices if np.linalg.det(matrix) != 0]
        return invertible_matrices
    
    @staticmethod
    def __generate_invertible(rank):
        """
        Generate all invertible matrices
        :param rank: side length of the square matrix
        :return:
        """
        return filter_invertible_matrices(generate_binary_matrices((rank,rank)))




if __name__ == "__main__":
    size = 3
    # m = np.random.randint(0,2,(3,3),dtype=int)
    m = np.identity(size,dtype=int)
    B = Best_basis(m)
    B.find_best()
    print(B.scores)
    
    [print(i,"\n") for i in B.all_best]
    # FIXME: these are all the same matrix that can be reached with elementary row and column operations,
    #  maybe instead of checking if something is invertible, we can generate matrices that are only different in the nr of elements
    #  and then filter those before constructing all same ones with elementary row operations
    #  we could then also take the transpose of the initial invertible matrices and do the same

    # FIXME: I wonder if we really need to compute GM to know if M' will be our preferred basis.
    #  Given the matrix multiplication, we can deduce which elements will be operated on and maybe we can know
    #  if there will will be a good separation of spins beforehand