from locale import DAY_6
import matplotlib.pyplot as plt
import array
from itertools import product
from re import I
import numpy as np
import small_mcm_tools as tools
import ebo_best_basis
# goal here is to construct 2 toy classes on a very small grid where we can do exhaustive search for the MCM
# we set some pixels to 0 or set a configuration that is not observed in the test set to 0
# we want to know what happens what we can still say about the shape we observe if we e.g., cut out a pixel that is 0 in the test data

# find best IM




# find best MCM

# write test scenarios


    
if __name__ == "__main__":
    m = tools.generate_class_dataset("c")
    # [print(i,"\n") for i in m]
    # print(np.sum(m,axis=0))a

    data_c = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("c")]
    data_t = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("t")]
    basis_c = ebo_best_basis.find_best_basis(9,data_c)
    basis_t = ebo_best_basis.find_best_basis(9,data_t)
    mcm_c = None
    mcm_t = None
    
    print(basis_c,"\n",basis_t)




            

        
