# """
# SPIN TOOLS

# ------------------------------
# author: Ebo Peerbooms
# contact: e.peerbooms@uva.nl 
# ------------------------------

# algo_best_basis.py 

# algorithm for finding most biased (basis) set of 
# independent spin operators

# """

import numpy as np 
import ebo_tools as tools


def get_most_biased_op(bias, basis):

	# most biased operator 

	most_biased_op = np.argmax(bias) + 1
	basis.append(most_biased_op)
	bias[most_biased_op - 1] = 0

	return bias, basis

def add_dependent_ops(op1, bias, op_list, dependent_ops):

	for op2 in op_list:

		op12 = np.bitwise_xor(op1, op2)

		if op12 not in dependent_ops:

			bias[op12 - 1] = 0
			dependent_ops.append(op12)

	return bias, dependent_ops

def find_best_basis(n, data):

    """
    function that finds best independent basis 

    n: number of variables
    obs: observables (include identity)

    """

    obs = tools.observables(n, data) # shape 1d of len 2^n


    # initialize basis
    basis = []
    dependent_ops = []

    # get absolute bias
    bias = np.abs(obs[1:])

    bias, basis = get_most_biased_op(bias, basis)

    while len(basis) < n:

        bias, basis = get_most_biased_op(bias, basis)

        last_op = basis[-1]

        bias, dependent_ops = add_dependent_ops(last_op, bias, dependent_ops, dependent_ops)
        bias, dependent_ops = add_dependent_ops(last_op, bias, basis[:-1], dependent_ops)

    return np.array(basis)


if __name__ == "__main__":

    seed = 253
    rng = np.random.default_rng(seed)
    n = 5
    s_dataset = rng.integers(2,size=(10,n))
    s_dataset = np.where(s_dataset == 0, -1, 1)
    
    res = find_best_basis(n, data=s_dataset)
    # print(res)
    # rhs = np.zeros(s_dataset.shape)
    # rhs[-1]

    # kron = tools.nkron(5,-1)
    # print(kron)


