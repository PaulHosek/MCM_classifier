from locale import DAY_6
from tkinter import N
import matplotlib.pyplot as plt
import array
from itertools import product
from re import I
import numpy as np
# from repositories.bsc.src.classify import MCM_Classifier
import small_mcm_tools as tools
import ebo_best_basis
import numpy as np


# goal here is to construct 2 toy classes on a very small grid where we can do exhaustive search for the MCM
# we set some pixels to 0 or set a configuration that is not observed in the test set to 0
# we want to know what happens what we can still say about the shape we observe if we e.g., cut out a pixel that is 0 in the test data

# find best IM


n_categories = 2
n_variables = 9
def main():
    classifier = MCM_Classifier(
        n_categories, n_variables, "placeholder", "placeholder")

    train_data = None
    train_labels = None

# find best MCM

# write test scenarios


    
if __name__ == "__main__":
    m = tools.generate_class_dataset("c")
    # [print(i,"\n") for i in m]
    # print(np.sum(m,axis=0))a

    # data_c = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("c")]
    # data_t = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("t")]
    # basis_c = ebo_best_basis.find_best_basis(9,data_c)
    # basis_t = ebo_best_basis.find_best_basis(9,data_t)

    c_binary_data = tools.generate_class_dataset("c")
    t_binary_data = tools.generate_class_dataset("t")
    train_c_d = c_binary_data[:60]
    test_c_d = c_binary_data[50:]# some overlap so we can see how the labels are affected by smoothing

    np.savetxt('INPUT/data/train_c_data.csv', np.vstack(c_binary_data[:60]), delimiter=',')
    np.savetxt('INPUT/data/train_t_data.csv', np.vstack(t_binary_data[:60]), delimiter=',')
    


    mcm_c = None
    mcm_t = None

    # Save test data
    test_c_d = c_binary_data[50:]
    test_t_d = t_binary_data[50:]

    np.savetxt('INPUT/data/test_data.csv', np.vstack((test_c_d, test_t_d)), delimiter=',')

    # Save labels
    test_c_labels = np.zeros(len(test_c_d))
    test_t_labels = np.ones(len(test_t_d))

    np.savetxt('INPUT/data/test_labels.csv', np.concatenate((test_c_labels, test_t_labels)), delimiter=',')




            

        
