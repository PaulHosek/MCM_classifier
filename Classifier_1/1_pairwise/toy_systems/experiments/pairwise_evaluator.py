import numpy as np
from numba import jit
import sys
sys.path.append("../")


# not sure if should inherent from fitter

# FIXME convention transform needed for new states, maybe make separete .dat that is the right convention
    # probably will need to sample from it later or calculte the evidence or whatever, then need this.
# FIXME check if formulas are correct using the convention, if use -1 1 state then should just use the ones from clelia


class Pairwise_evaluator():
    def __init__(self, paramter_path:str,nspins:int):
        self.parameter_path = paramter_path
        self.nspins = nspins
        self.fields = np.zeros(nspins,dtype=np.double)
        self.couplings = np.zeros(int(nspins*(nspins-1)/2),dtype=np.double)


    def load_ising_paramters(self):
        """
        Read the .j file of the inferred ising/potts paramters.

        """
        # loads the field and coupling paramters
        # be careful with the indexing on this stuff
        all_param = np.loadtxt(self.parameter_path)
        self.__validate_jfile(all_param)
        self.fields[:] = all_param[:self.nspins]
        self.couplings[:] = all_param[self.nspins:]



    def __validate_jfile(self, res):
        """Validate dimensions of the potts paramter ".j" file."""
        all_param = np.loadtxt(self.parameter_path)
        N = self.nspins
        if len(all_param.shape) != 1:
            raise ValueError(f"Input data in is not 1-dimensional. Input file path: {self.parameter_path}")
        if all_param.shape[0] != N + N*(N-1)/2:
            raise ValueError(f"Nr of total paramters, shape ({all_param.shape}), dim0 does not match expected {N + N(N-1)/2} samples based on {N} spins.")

    def calc_energy(self, state): # state assumed to be in convention -1 1
        h = self.__calc_fields(self.fields, state)
        j = self.__calc_couplings(self.couplings,state)
        return -1*(h+j)


    @staticmethod
    @jit("(float64[:], int64[:])", nopython=True) # state assumed to be in convention -1 1
    def __calc_fields(fields, state) -> np.double:
        return np.sum(fields * state)    # FIXME check if this is correct based on the convention

    @staticmethod
    def __calc_couplings(couplings, state)->np.double:
        """Compute -1*sum(J_ij*s[i]*s[j]) for all (i,j) with i < j.
        Outer product and masking triangle.
        Could maybe make this faster by saving the multiplications where any s is 0.
        """
        return np.sum(couplings*((state[:,None]*state)[~np.tri(len(state),dtype=bool)]))
    
    # numba this as well

    @staticmethod
    def unpackbits2d(x, num_bits):
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])[:,::-1]

    def partiton_function(self):
        # generate the value for Z
        # this is the most expensive, only do once -> class
        # calcualte portion we observed and assign mass there, all other ones we do not need to generate since all the same
            # assign their value proportionally to their part of 2^N states

        # 1. Calculate energy for every observed state

        # 1. generate every possible state
        assert self.nspins <= 15, "> 15 spins. Avoid calculating Z."
        all_states = self.unpackbits2d(np.arange(2**self.nspins), self.nspins)
        # 2. calculate energy or each state
        all_E = np.apply_along_axis(self.calc_energy,1,all_states)


        return res


        # 3. take exp of - energy of every state and sum the results up over all states

        # get Z out




    def predict(state):
        # assigns probability to a state using formula 1 from ACE Barton et al. paper.
        pass

if __name__ == "__main__":
    spin4_path = "../output_small/4spin/4spin_sep-output-out.j"
    mod = Pairwise_evaluator(spin4_path,4)
    mod.load_ising_paramters()
    Z = mod.partiton_function()
    print(Z)