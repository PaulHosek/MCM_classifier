import os
import sys

sys.path.append("../")
import src.paper_utils as utils
import argparse


def main(sample_s,split_letter):
    # full sample 6315: train + test
    n_categories = 10
    n_variables = 121
    all_data_path="../INPUT_all/data/combined_split_{}".format(split_letter)
    result_samples_dir="../OUTPUT/sample_sizes_split_{}".format(split_letter)
    data_path="../INPUT/data"
    mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
    data_filename_format = "full-images-unlabeled-{}.dat"
    communities_path = "../OUTPUT/comms/"
    seed = 42 # always same seed

    clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path,
                communities_path]  # data path here for subsampling

    for s in range(50):

        utils.evaluate_subsample(sample_s, clf_args, all_data_path=all_data_path,
                                 result_sample_sizes_dir=result_samples_dir, fname_start="half-", seed=seed,
                                 input_data_path=data_path)


def parse_arguments():
    """Parses command-line arguments and returns a dictionary containing them.

  Returns:
    A dictionary containing parsed arguments with keys matching the flag names.
  """
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--sample_s", type=int, required=True,
                        help="Sample size.")
    parser.add_argument("--split_letter", type=str, required=True,help="A or B")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.sample_s, args.split_letter)
    # python3 run_fullsample.py --sample_s 10
