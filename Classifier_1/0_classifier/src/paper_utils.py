
import numpy as np

from src.classify import MCM_Classifier
import json
import os
from shutil import copytree




### --- FUNCTIONS FOR DATA SUBSETTING ---
def evaluate_subsample(sample_size,MCM_Classifier_init_args, all_data_path="../INPUT_all/data",
                        result_sample_sizes_dir="../OUTPUT/sample_sizes", comms_dir = "../OUTPUT/comms",estimator="add_smooth", seed=None,fname_start="train-"):
    """
    Generate sample_size number of samples and populate "../INPUT" folder. 
    Then fit the model to that data and save MCM and Counts from that model
      in a directory named after the sample size in the "../OUTPUT/sample_sizes" folder.

    :param sample_size: The number of images per class that should be used, if None then use all..
    :type sample_size: int
    :param all_data_path: The path to the data directory that will not be changed and where data is read from,
                            defaults to "../INPUT_all/data"
    :type all_data_path: str, optional
    :param result_sample_sizes_dir: The path to the output directory for saving the results,
                            defaults to "../OUTPUT/sample_sizes"
    :type result_sample_sizes_dir: str, optional
    :param comms_dir: directory of the communities after the current fitting
    """
    # subsample the data

    subsample_data(sample_size, all_data_path=all_data_path, seed=seed, fname_start=fname_start)
    # Fit new classifier object
    classifier = MCM_Classifier(*MCM_Classifier_init_args)
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000, estimator=estimator)


    # Save MCMS and Counts
    nwdir = os.path.join(result_sample_sizes_dir, str(sample_size))
    os.makedirs(nwdir, exist_ok=True)

    mcmdir = os.path.join(nwdir, "MCMs")
    countsdir = os.path.join(nwdir, "Counts")
    os.makedirs(mcmdir, exist_ok=True)
    os.makedirs(countsdir, exist_ok=True)


    # Append the number of files + 1 to the file names
    mcm_file_name = "MCMs_" + str(len(os.listdir(mcmdir))) + ".json"
    counts_file_name = "Counts_" + str(len(os.listdir(countsdir))) + ".json"

    # Save MCMS and Counts with the updated file names
    with open(os.path.join(mcmdir, mcm_file_name), 'w') as f:
        json.dump([arr.tolist() for arr in classifier.get_MCMs()], f, indent=2) 

    with open(os.path.join(countsdir, counts_file_name), 'w') as f:
        json.dump(classifier.get_Counts(), f, indent=2)


    # # Copy the new communities -> are also in MCM now
    # ncom = os.path.join(nwdir, "comms")
    # os.makedirs(ncom,exist_ok=True)
    # copytree(comms_dir, ncom,dirs_exist_ok=True)




    


def subsample_data(sample_size, all_data_path="../INPUT_all/data", input_data_path="../INPUT/data", seed=42,fname_start = "train-"):
    """Clear the input_data_path folder and fill it with samples from the all_data_path folder.
    
    :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

    """
    rng = np.random.default_rng(seed)

    # Iterate over the files and delete the ones that start with "train-"
    for file in os.listdir(input_data_path):
        if file.startswith(fname_start):
            os.remove(os.path.join(input_data_path, file))

    # generate new input data 
    for file in os.listdir(all_data_path):
        if file.startswith(fname_start):
            inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
            np.savetxt(os.path.join(input_data_path, file), rng.choice(inp, sample_size,replace=False), fmt="%s")

#------------------------------ 


# def nudge_dataset(X, Y):
#     """
#     This produces a dataset 5 times bigger than the original one,
#     by moving the 8x8 images in X around by 1px to left, right, down, up
#     """
#     direction_vectors = [
#         [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
#     ]

#     def shift(x, w):
#         return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

#     X = np.concatenate(
#         [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
#     )
#     Y = np.concatenate([Y for _ in range(5)], axis=0)
#     return X, Y
