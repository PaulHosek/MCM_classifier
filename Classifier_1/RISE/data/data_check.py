import numpy as np 
from spin_tools import tools

n = 16



data = np.loadtxt('20190828_binsec1.dat', dtype=str)
data = [int(s,2) for s in data]
u, c = np.unique(data, return_counts = True)

print(np.size(u)/2**n)