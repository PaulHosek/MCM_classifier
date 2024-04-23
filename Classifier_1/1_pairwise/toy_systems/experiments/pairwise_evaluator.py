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
        self.all_E = np.empty(1)
        self.Z = 0.0
        self.all_states = np.empty(1)


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
        return h+j


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

    def calc_partitionf(self):
        """Compute the partition function's value. Needs 2**n_spins*nspins*sizof(int) space in memory.
        Also sets values for self.all_states, self.all_E, self.Z, self.all_P.
        :return: self.Z The value of the parition function
        :rtype: float
        """
        assert self.nspins <= 15, "> 15 spins. Avoid calculating Z."
        self.all_states =  self.unpackbits2d(np.arange(2**self.nspins), self.nspins)
        self.all_E = np.apply_along_axis(self.calc_energy,1,self.all_states)
        self.Z = np.sum(np.exp(-self.all_E))
        self.all_P = np.exp(-self.all_E)/self.Z
        return self.Z

    def predict_with_Z(self,state):
        """Get the probability of a test state, given the parition function.

        :param state: 1d binary np array of the state with dtype int-like.
        :type state: np.array 1d
        :return: P(state|Model)
        :rtype: float
        """
        # assigns probability to a state using formula 1 from ACE Barton et al. paper.
        assert self.all_states.ndim == 2, "self.all_states is not 2d. Didi you call the calc_partition_function?"
        assert isinstance(state, (np.ndarray, list)), "input state must be a numpy array or a list"
        # assert state.ndim == 1, f"input state must be 1-dimensional, currently: state.ndim =  {state.ndim}"
        idx = int("".join(state.astype(str)),base=2)
        return self.all_P[idx]

    @staticmethod
    def unpackbits2d(x, num_bits):
        """Unpacks an 1d array of integers into a 2d array of their binary represention.

        :param x: 1d NP array of integers
        :type x: np.array 1d
        :param num_bits: number of spins/ binary size of the largest potential integer
        :type num_bits: int
        :raises ValueError: If data type of the np array is not int-like.
        :return: 2d array of size (x,num_bits)
        :rtype: np.array
        """
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])[:,::-1]
    

    def spin_avgs(self):
        # for every spin we need to multiply its value with the probability of the state
        #
        return np.sum(self.all_P[:,None] *self.all_states, axis=0)

    def correls(self):
        pass

        
if __name__ == "__main__":
    spin4_path = "../output_small/4spin/4spin_sep-output-out.j"
    mod = Pairwise_evaluator(spin4_path,4)
    mod.load_ising_paramters()
    mod.calc_partitionf()
    si = mod.spin_avgs()
    print(si)








    # P = mod.predict_with_Z(np.array([1,0,0,1]))
    # res = np.sum([mod.predict_with_Z(i) for i in mod.all_states])

    # res = 0
    # for i in mod.all_states:
    #     x = mod.predict_with_Z(i)
    #     res += x
    #     print(x)

    # print("\nSum over all states (should be 1): ",res)