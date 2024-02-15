import numpy as np

from src.loaders import load_data
import json
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
