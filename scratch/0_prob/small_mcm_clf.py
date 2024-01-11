from ast import LShift
import matplotlib.pyplot as plt
import array
from itertools import product
import numpy as np
import sys
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier
from src.plot import plot_confusion_matrix, plot_label_prob_diff


import small_mcm_tools as tools
import ebo_best_basis
import numpy as np


# goal here is to construct 2 toy classes on a very small grid where we can do exhaustive search for the MCM
# we set some pixels to 0 or set a configuration that is not observed in the test set to 0
# we want to know what happens what we can still say about the shape we observe if we e.g., cut out a pixel that is 0 in the test data

# find best IM


n_categories = 2
n_variables = 9
mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
data_filename_format = "train-images-unlabeled-{}_bootstrap.dat"

def main():
    print("{:-^50}".format("  MCM-Classifier  ")) 

    test_data = load_data("INPUT/data/test-images-unlabeled-all-uniform.txt").astype(int)
    test_labels = load_labels("INPUT/data/test-labels-uniform.txt").astype(int)
    # Step 1: Initialize classifier
    print(test_data)
    classifier = MCM_Classifier(
    n_categories, n_variables, mcm_filename_format, data_filename_format
    )

    # Step 2: Train
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)
    # classifier.init()

    # Step 3: Evaluate
    predicted_classes, probs = classifier.evaluate(test_data, test_labels)

    # Step 4: Save classification report and other stats
    report = classifier.get_classification_report(test_labels)
    # train_data = None
    # train_labels = None

# find best MCM

# write test scenarios



def iter_flat(ls): 
    return np.array([i.flatten() for i in ls]) 
    
if __name__ == "__main__":
    np.savetxt("INPUT/MCMs/train-images-unlabeled-0_bootstrap_comms.dat",np.identity(n_variables))
    np.savetxt("INPUT/MCMs/train-images-unlabeled-1_bootstrap_comms.dat",np.identity(n_variables))
    # Generate training and test datasets
    c_binary_data = tools.generate_class_dataset("c")
    t_binary_data = tools.generate_class_dataset("t")
    np.savetxt("INPUT/data/train-images-unlabeled-0.dat", iter_flat(c_binary_data[:60]), delimiter='', fmt='%i',)
    np.savetxt('INPUT/data/train-images-unlabeled-1.dat', iter_flat(t_binary_data[:60]), delimiter='', fmt='%i',)
    
    test_c_d = iter_flat(c_binary_data[50:])
    test_t_d = iter_flat(t_binary_data[50:])
    test_data = np.concatenate([test_c_d,test_t_d])
    test_c_labels = np.zeros(len(test_c_d))
    test_labels = np.concatenate((np.ones(len(test_c_d),dtype=int), np.zeros(len(test_t_d),dtype=int)))

    np.savetxt('INPUT/data/test-images-unlabeled-all-uniform.txt', test_data, delimiter='', fmt='%i',)
    np.savetxt('INPUT/data/test-labels-uniform.txt', test_labels, delimiter='', fmt='%i',)
   
    main()



















    # [print(i,"\n") for i in m]
    # print(np.sum(m,axis=0))a

    # data_c = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("c")]
    # data_t = [tools.arr_to_int(i.flatten()) for i in tools.generate_class_dataset("t")]
    # basis_c = ebo_best_basis.find_best_basis(9,data_c)
    # basis_t = ebo_best_basis.find_best_basis(9,data_t)


            

        
