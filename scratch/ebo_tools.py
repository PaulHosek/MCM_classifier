import numpy as np

def observables(n, data, x = -1, return_pdata = False):

	"""
	description

	--- parameters ---

	--- optional parameters ---

	--- returns ---

	"""

	data = np.array(data)

	# check data type 
	if data.dtype not in ['int32','int64']:
		print('Type error: Please provide data in integer format')
		return

	fn = nkron(n,x)

	pdata = np.zeros(2**n)

	unique, counts = np.unique(data, return_counts = True)

	pdata[unique] = counts/np.sum(counts)

	obs = np.dot(fn.T, pdata)

	if return_pdata:

		return obs, pdata

	else: 
		
		return obs

def kron_mat(n, x = -1, convention = 'XOR'):

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