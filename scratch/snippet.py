import numpy as np 

n = 4

# all 2^n operators including identity element 0
all_ops = np.arange(2**n)

# all observables (mock)
all_obs = np.random.uniform(-1,1,2**n)


# the identity always has expectation value 1
all_obs[0] = 1

# initialize basis
basis = []
dependent = []

# go up to n + 1 to avoid having to treat identity separately
while len(basis) < n + 1:

	# find most biased operator
	op = np.argmax(all_obs)
	basis.append(op)

	# add all new dependencies
	dependent += [i ^ op for i in basis] + [i ^ op for i in dependent]
	
	# remove duplicates
	dependent = list(set(dependent))

	# remove dependent + basis from observables
	all_obs[np.array(dependent)] = 0

print(basis)
print(dependent)