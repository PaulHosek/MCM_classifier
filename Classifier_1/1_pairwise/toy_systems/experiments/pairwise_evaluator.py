import numpy as np
import sys
sys.path.append("../")


# not sure if should inherent from fitter


class Pairwise_evaluator():
    def __init__(self, paramter_path:str,nspins:int):
        self.parameter_path = paramter_path
        self.nspins = nspins
        self.fields = np.zeros(nspins)
        self.couplings = np.empty(int(nspins*(nspins-1)/2))


    def load_ising_paramters(self):
        """
        Read the .j file of the inferred ising/potts paramters.

        """
        # loads the field and coupling paramters
        # be careful with the indexing on this stuff
        all_param = np.loadtxt(self.parameter_path)
        self.__validate_jfile(all_param)
        self.fields = all_param[:self.nspins]
        self.couplings = all_param[self.nspins:]

    def __validate_jfile(self,all_param):
        """Validate dimensions of the potts paramter ".j" file."""
        all_param = np.loadtxt(self.parameter_path)
        N = self.nspins
        if len(all_param.shape) != 1:
            raise ValueError(f"Input data in is not 1-dimensional. Input file path: {self.parameter_path}")
        if all_param.shape[0] != N + N*(N-1)/2:
            raise ValueError(f"Nr of total paramters, shape ({all_param.shape}), dim0 does not match expected {N + N(N-1)/2} samples based on {N} spins.")



    # numba this shit
    def energy(state):
        h = fields(state)
        pass

    # number here too
    def fields(state):
        # make this jittable
        # calcualte h(xi) and sum over them
        # 1. find the right xi in the fields datastructure
        # 2. sum them up    
        pass

    # numba here too
    def couplings(state):

        pass

    # numba this as well
    def partiton_function():
        # generate the value for Z
        # this is the most expensive, only do once -> class
        # calcualte portion we observed and assign mass there, all other ones we do not need to generate since all the same
            # assign their value proportionally to their part of 2^N states
        pass


    def predict(state):
        # assigns probability to a state using formula 1 from ACE Barton et al. paper.
        pass

if __name__ == "__main__":
    spin4_path = "../output_small/4spin/4spin_sep-output-out.j"
    mod = Pairwise_evaluator(spin4_path,4)
    mod.load_ising_paramters()