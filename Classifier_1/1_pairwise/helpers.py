import os
import numpy as np











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
