import os
import numpy as np
import shutil



def subsample_cat_ace(sample_size,fname, all_data_path="./INPUT_all/data/", input_data_path="./INPUT/data/", seed=42):
    """Clear input_data_path and fill it, from the all_data_path, with a single folder of the category we are interested in.
    
    :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

    """
    rng = np.random.default_rng(seed)

    clear_cat(input_data_path, fname)
    # generate new input data 
    for file in os.listdir(all_data_path):
        if file.startswith(fname):
            print(file)
            inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
            subfolder_name = file.split(".")[0]  
            subfolder_path = os.path.join(input_data_path, subfolder_name) 

            os.makedirs(subfolder_path, exist_ok=True) 
            arr = rng.choice(inp, sample_size, replace=False)
            arr = np.append(arr, ["0"*121, "1"*121])
            np.savetxt(os.path.join(subfolder_path, file), arr, fmt="%s")

def clear_cat(path, folder_prefix):
    """In path, delete all folders that begin with folder_prefix."""
    for folder in os.listdir(path):
        if folder.startswith(folder_prefix):
            folder_path = os.path.join(path, folder)
            shutil.rmtree(folder_path)
    
if __name__ == "__main__": 
    subsample_cat_ace(10,"train-images-unlabeled-0")
    
    
# def clear_directory(path):
#     """Deletes all files and subdirectories in a given directory.

#     :param path: Path to the directory to clear.
#     """
#     ford filename in os.listdir(path):
#         file_path = os.path.join(path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')

