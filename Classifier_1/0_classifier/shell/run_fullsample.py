import sys

sys.path.append("../")
import src.paper_utils as utils
import argparse


def main(sample_s, all_data_path="../INPUT_all/data/combined_data",
         result_samples_dir="../OUTPUT/sample_sizes_fromfull", data_path="../INPUT/data"):
    # full sample 6315: train + test
    n_categories = 10
    n_variables = 121
    mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
    data_filename_format = "full-images-unlabeled-{}.dat"
    communities_path = "../OUTPUT/comms/"

    clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path,
                communities_path]  # data path here for subsampling

    for s in range(10):
        seed = s + int(sample_s)
        utils.evaluate_subsample(sample_s, clf_args, all_data_path=all_data_path,
                                 result_sample_sizes_dir=result_samples_dir, fname_start="full-", seed=seed,
                                 input_data_path=data_path)


def parse_arguments():
    """Parses command-line arguments and returns a dictionary containing them.

  Returns:
    A dictionary containing parsed arguments with keys matching the flag names.
  """
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--sample_s", type=int, required=True,
                        help="Sample size.")
    parser.add_argument("--all_data_path", type=str, required=False,
                        help="Path to the directory to subsample from.")
    parser.add_argument("--result_samples_dir", type=str, required=False,
                        help="Path to the directory to store MCM and Count.json.")
    parser.add_argument("--data_path", type=str, required=False,
                        help="Path to input folder where to store subsamples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.sample_s, args.all_data_path, args.results_samples_dir, args.data_path)
    # python3 run_fullsample.py --sample_s 10
