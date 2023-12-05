import numpy as np
from collections import defaultdict

from transform.gauge_transform_algs import gauge_transform_xor
from inversion.invertible_matrices_algs import generate_invertible



# build all possible gauge transforms for this size
# apply all gauge transforms
# find the one that is

class best_basis():
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
        self.best_g_score = np.min(list(self.scores.keys()))
        self.all_best = self.scores[self.best_g_score]
        self.best_g = self.all_best[0]

    def __score_bases(self, m) -> None:
        """
        Generate and score alternative bases.
        Do GM = M' for all g in {g}.

        :param m: square binary matrix
        :type m: _type_
        """
        assert set(m) <= {0,1}, "values outside 0, 1"
        assert m.shape[0] == m.shape[1], "m is not square"
        assert len(m.shape) == 2, "m is not 2d"
        
        t = generate_invertible(len(m))
        t2 = np.empty(shape = (len(t), size, size))

        for i, g_i in enumerate(t):
            m2 = np.array(gauge_transform_xor(m, g_i))
            t2[i,:,:] = m2

            score  = self.__scoring_sum(m2)
            self.scores[score].append(m2) # FIXME: Note: Placeholder scoring function.

    @staticmethod
    def __scoring_sum(m2) -> float:
        return np.sum(m2)
       
        


if __name__ == "__main__":
    size = 3
    # m = np.random.randint(0,2,(3,3),dtype=int)
    m = np.identity(size,dtype=int)
    B = best_basis(m)
    B.find_best()
    
    [print(i,"\n") for i in B.all_best]
    # FIXME: these are all the same matrix that can be reached with elementary row and column operations,
    #  maybe instead of checking if something is invertible, we can generate matrices that are only different in the nr of elements
    #  and then filter those before constructing all same ones with elementary row operations
    #  we could then also take the transpose of the initial invertible matrices and do the same

    # FIXME: I wonder if we really need to compute GM to know if M' will be our preferred basis.
    #  Given the matrix multiplication, we can deduce which elements will be operated on and maybe we can know
    #  if there will will be a good separation of spins beforehand