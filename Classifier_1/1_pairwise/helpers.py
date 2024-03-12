import os
import numpy as np




def subsample_data_ace(sample_size, all_data_path="../INPUT_all/data", input_data_path="../INPUT/data", seed=42,fname_start = "train-"):
    """Clear the input_data_path folder and fill it with samples from the all_data_path folder.
        Note: Every sample will be getting its own subfolder in this "ace" version.
    
    :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

    """
    rng = np.random.default_rng(seed)

    clear_directory(input_data_path)

    # generate new input data 
    for file in os.listdir(all_data_path):
        if file.startswith(fname_start):
            inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
            subfolder_name = file.split(".")[0]  # extract the file name without extension
            subfolder_path = os.path.join(input_data_path, subfolder_name)  # create subfolder path

            os.makedirs(subfolder_path, exist_ok=True)  # create subfolder if it doesn't exist
            np.savetxt(os.path.join(subfolder_path, file), rng.choice(inp, sample_size, replace=False), fmt="%s")


def clear_directory(path):
    """Deletes all files and subdirectories in a given directory.

    :param path: Path to the directory to clear.
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

