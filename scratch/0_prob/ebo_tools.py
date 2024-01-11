import numpy as np

def observables(n, data, x = -1, return_pdata = False):

	"""
	description

	--- parameters ---

	--- optional parameters ---

	--- returns ---

	"""

	data = np.array(data)

	# 32x32 array with -1 or 1 * 32x32 array with first entry 0.58 and last entry 0.42
	# so we sum 2 entries 

	# check data type 
	if data.dtype not in ['int32','int64']:
		print('Type error: Please provide data in integer format')
		return

	# assume input n=2
	fn = nkron(n,x) # 2^n x 2^n matrix: here 4x4

	pdata = np.zeros(2**n) # why do we need this big of an array? the next line can only give -1 and 1 as counts. len 2 should be enough.

	unique, counts = np.unique(data, return_counts = True) # [-1  1] [10 10]
	pdata[unique] = counts/np.sum(counts) # the second and last entry are only ones not 0
	obs = np.dot(fn.T, pdata) 

	# print(unique,counts)
	# print(fn.shape)
	# print("fn.T\n",fn.T)
	# print()
	# print("pdata\n",pdata)
	# print()

	# print(obs)



	if return_pdata:

		return obs, pdata

	else: 
		
		return obs

def kron_mat(n, x = -1, convention = 'XOR'): # i think: returns all possible spin configurations for an nxn matrix

	"""
	function that returns spin configuration
	matrices for a given n recursively

	note: doesn't work for n >~ 15 due to memory issues

	--- parameters ---

	n: number of variables 

	--- optional parameters ---

	x: spin convention: -1 or 0 

	--- returns ---

	f: (2^n,2^n) spin matrix

	"""

	if x == -1:
		if convention == 'XOR':
			f = np.array([[1,1],[1,-1]])
		elif convention == 'OLD': 
			f = np.array([[1,-1],[1,1]])
		else:
			return ValueError('convention must by XOR or OLD.')
	elif x == 0:
		f = np.array([[1,0],[1,1]])
	else:
		raise ValueError('choose x = 0 or x = -1.')

	if n == 1:

		return f 

	else: 
		# print(f, "\n",np.kron(f, nkron(n - 1, x)))
		# print()
		return np.kron(f, nkron(n - 1, x))

def nkron(n, x = -1, return_inverse = False):

	"""
	description

	--- parameters ---

	--- optional parameters ---

	--- returns ---

	"""


	fn = kron_mat(n, x)

	if return_inverse:

		return fn, (np.linalg.inv(fn)) # this doesn't work

	else:

		return fn