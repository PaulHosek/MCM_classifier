# for every k
# for every digit, fit an MCM on the whole train data


import numpy as np
import os




sys.path.append("../")
import src.paper_utils as utils
import argparse



def classify(sample_s, ):
      
    seed = 1
    n_categories = 10 
    n_variables = 121  
    mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
    data_filename_format = "train-images-unlabeled-{}.dat"
    data_path = "../INPUT/data/"
    communities_path = "../OUTPUT/comms/"
    output_path = "../OUTPUT/"

    all_data_path="../INPUT_all/data"
    result_sample_sizes_dir = "../OUTPUT/sample_sizes"
    clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path] # data path here for subsampling

    utils.evaluate_subsample(sample_s, clf_args, seed=seed, seed_plus=True)


n = 11
dirname = "../../data/OUTPUT/mcm/fourier/"
k = 1

n = 11


for k in items_above_diagonal(n): # take fair samples
    k_path = os.path.join(dirname, f"{k}")
    os.makedirs(k_path, exist_ok=True)
    for digit in range(10):
        # train data
        img = np.genfromtxt(os.path.join(k_path,f"train-images-unlabeled-{digit}.dat"),delimiter=1, dtype=int)

        # freq = np.apply_along_axis(lambda x: binarize_image(lowpass_zigzag1d(x, k)).astype(int), axis=1, arr=img)
        # np.savetxt(os.path.join(k_path,f"fourier-images-unlabeled-{digit}.dat"),freq,fmt="%d", delimiter="")
        raise KeyboardInterrupt
