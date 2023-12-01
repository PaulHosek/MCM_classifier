import numpy as np


## Loaders
def load_data(path: str):

    binary_strings= np.loadtxt(path, dtype=str)
    arrays_list = [np.fromiter(s, dtype=int) for s in binary_strings]
    return arrays_list

def load_labels(path: str):
    return np.loadtxt(path,dtype=str)

def load_mcm(path: str):
    return np.loadtxt(path, dtype=str)

## Helpers


def generate_bootstrap_samples(data: np.ndarray, bootstrap_name: str, n_samples: int):
    bootstrap_samples = np.random.choice(data,size=n_samples,replace=True)

    zero_sample = np.zeros_like(bootstrap_samples[0], dtype=int)
    one_sample = np.ones_like(bootstrap_samples[0], dtype=int)

    bootstrap_samples = np.concatenate([bootstrap_samples, zero_sample, one_sample])


    np.savetxt(f"INPUT/data/{bootstrap_name}.dat", bootstrap_samples, fmt="%d", delimiter="")
    print("Bootstrap samples generated and saved. \n")
