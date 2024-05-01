import os
import sys

sys.path.append("../")
import src.paper_utils as utils
import argparse



def main(sample_s):
      
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
##########3
  # full sample 6315: train + test
  

def parse_arguments():
    """Parses command-line arguments and returns a dictionary containing them.

  Returns:
    A dictionary containing parsed arguments with keys matching the flag names.
  """
    parser = argparse.ArgumentParser(description="run split programme")
    parser.add_argument("--sample_s", type=int, required=True,
                        help="Sample size.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.sample_s)
