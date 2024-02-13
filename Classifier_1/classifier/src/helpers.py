import numpy as np
from src.loaders import load_data
import json
from classify import MCM_Classifier
import os


def print_box(message: str) -> None:
    print("{:-^50}".format(""))
    print("{:-^50}".format("  " + message + "  "))
    print("{:-^50}".format(""))
    print()

def generate_bootstrap_samples(data_path: str,filename:str, bootstrap_name: str, n_samples: int):
    """Generates bootstrap samples from the provided data in the INPUT/data folder.
    Places them in the INPUT/data/bootstrap_name folder.
    """
    data = load_data(data_path + filename + ".dat")
    print("Generating bootstrap samples...")
    samples = []
    for i in range(n_samples):
        bootstrap_sample = data[np.random.randint(0, data.shape[0])]
        samples.append(bootstrap_sample)
    all_zeros = np.zeros(len(samples[0]), dtype=int)
    all_ones = np.ones(len(samples[0]), dtype=int)
    samples.append(all_zeros)
    samples.append(all_ones)
    np.savetxt(
        "{}/{}.dat".format(data_path, bootstrap_name),
        samples,
        fmt="%d",
        delimiter="",
    )
    print("Done!")



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


### --- FUNCTIONS FOR DATA SUBSETTING ---
def evaluate_subsample(sample_size,MCM_Classifier_init_args, all_data_path="../INPUT_all/data", result_sample_sizes_dir="../OUTPUT/sample_sizes"):
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
    """
    # subsample the data
    subsample_data(sample_size, all_data_path=all_data_path)
    # Fit new classifier object
    classifier = MCM_Classifier(*MCM_Classifier_init_args)
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)


    # Save MCMS and Counts
    nwdir = os.path.join(result_sample_sizes_dir, f"{sample_size}")
    os.makedirs(nwdir, exist_ok=True)

    with open(os.path.join(nwdir, "MCMs.json"), 'w') as f:
        json.dump([arr.tolist() for arr in classifier.get_MCMs()],f, indent=2) 

    with open(os.path.join(nwdir, "Counts.json"), 'w') as f:
        json.dump([j.tolist() for i in classifier.get_Counts() for j in i],f, indent=2)


def subsample_data(sample_size=None, all_data_path="../INPUT_all/data", input_data_path="../INPUT/data", seed=42):
    """Clear the input_data_path folder and fill it with samples from the all_data_path folder.
    
    :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

    """
    rng = np.random.default_rng(seed)

    # Iterate over the files and delete the ones that start with "train-"
    for file in os.listdir(input_data_path):
        if file.startswith("train-"):
            os.remove(os.path.join(input_data_path, file))

    # generate new input data 
    for file in os.listdir(all_data_path):
        if file.startswith("train-"):
            inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
            np.savetxt(os.path.join(input_data_path, file), rng.choice(inp, sample_size), fmt="%s")

#------------------------------ 