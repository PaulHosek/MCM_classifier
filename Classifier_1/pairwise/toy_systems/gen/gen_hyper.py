import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt


class Hyperspinmodel(hnx.Hypergraph):
    def __init__(self, incidence_dict, edge_weights, convention=0):
        """
        Initializes a Hyperspinmodel object.

        :param incidence_dict: A dictionary representing the incidence structure of the hypergraph.
        :type incidence_dict: dict
        :param edge_weights: A list of edge weights corresponding to the edges in the hypergraph.
        :type edge_weights: list
        :param convention: The convention to be used for representing spin values (0 or -1).
        :type convention: int
        """
        assert len(incidence_dict) == len(edge_weights), "Number of edge weights and edges are not the same."
        assert convention in [0, -1], "Convention must be integer -1 or 0"
        super().__init__(incidence_dict)
        self.nspin = len(list(self.nodes))
        self.edge_weights = edge_weights
        self.energy = np.empty(0)
        self.Z = np.nan
        self.probabilities = np.empty(0)
        self.convention = convention


    def compute_spinmodel(self):
        """
        Computes the energy distribution, partition function Z value, and probability distribution of the spin model.

        :return: None
        """
        all_states = self.gen_allbitstrings()
        all_states[all_states == 0] = self.convention  # FIXME. Test if works with -1 too.
        incm = self.incidence_matrix().toarray()  # spin (rows) x link (cols)
        weighted_incm = incm * self.edge_weights
        self.energy = (all_states[:, :, None] * weighted_incm).sum(axis=2).sum(axis=1)  # state spin values * interaction; sum over interactions
        self.Z = np.sum(self.energy)
        self.probabilities = self.energy / self.Z


    def gen_allbitstrings(self):
        """
        Generate a 2D numpy array of every possible bit string for a given number of bits/string length.
        Caution: Size of generated array grows exponentially.

        :return: A 2D numpy array of shape (2**num_bits, num_bits) containing all possible bit strings.
        :rtype: numpy.ndarray
        """
        x = np.arange(2 ** self.nspin, dtype=int)
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2 ** np.arange(self.nspin, dtype=x.dtype).reshape([1, self.nspin])
        return (x & mask).astype(bool).astype(int).reshape(xshape + [self.nspin])[:, ::-1]


if __name__ == "__main__":
    scenes = {
        1: ('A', 'B'),
        2: ('C', 'D', 'A', 'B'),
        3: ("B"),
        4: ("A"),
        5: ("B","C","D"),
    }
    seed = 0
    rng = np.random.default_rng(seed)
    edge_weights_Jij = rng.random(5)

    H = Hyperspinmodel(scenes,edge_weights_Jij)
    H.compute_spinmodel()
    print(H.probabilities)
    # plt.subplots(figsize=(5, 5))
    # hnx.draw(H)
    # plt.show()
