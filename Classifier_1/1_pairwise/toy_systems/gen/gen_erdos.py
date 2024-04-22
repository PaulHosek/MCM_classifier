import networkx as nx
import numpy as np


def random_subset(G, probability,seed):
  """
  Creates a new subset from the given array with a probability of inclusion 
  for each element.

  Args:
      G: The Graph array to create the subset from.
      probability: The probability (between 0 and 1) of including an element 
                   in the subset.

  Returns:
      A new NumPy array containing a random subset of elements from the original array.
  """
  rng = np.random.default_rng(seed)
  arr = np.array(G.edges)
  mask = rng.random(arr.shape[0]) < probability
  return arr[mask]

def gen_config(G,n,include_edge_p,seed):
    sample = np.unique(random_subset(G,include_edge_p,seed).flatten())
    bin_str = np.zeros(n,dtype=int)
    bin_str[sample] = 1
    return bin_str
   

if __name__ == "__main__":
    """Generate erdos-renyi graph and include each edge with probability given by include_edge_p in a state. Repeat this 10K times."""

    seed = 42
    rng = np.random.default_rng(seed)
    n = 8
    p = .3
    G = nx.erdos_renyi_graph(n,p)
    include_edge_p = 0.2



    states = np.array([gen_config(G,n,include_edge_p,cur_seed) for cur_seed in range(10000)])
    states[-1,:] = 0
    states[-2,:] = 1
    print(states.shape)
    np.savetxt("8_erdos.dat", states,fmt="%d", delimiter=" ")






    # fields = rng.normal(loc=0,scale=1,size=n)
    # coulings = rng.normal(loc=0,scale=5,size=int(n*(n-1)/2)) # same as in paper
    # print(nx.adjacency_matrix(G).todense())
    # now sample 10K possible pairs by including an edge with some probability every time



