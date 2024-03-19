import os
import sys

sys.path.append("../")
import src.paper_utils as utils
import argparse



def main(sample_s,split_letter):
      

  n_categories = 10 
  n_variables = 121  
  mcm_filename_format = "half-images-unlabeled-{}_bootstrap_comms.dat"
  data_filename_format = "half-images-unlabeled-{}.dat"
  data_path = "../INPUT/data/"
  communities_path = "../OUTPUT/comms/"

  clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path] 
  adp = "../INPUT_all/data/combined_split_{}".format(split_letter)
  rssd = "../OUTPUT/sample_sizes_split_{}".format(split_letter)
  for i in range(3):
      utils.evaluate_subsample(sample_s, clf_args,all_data_path=adp, result_sample_sizes_dir=rssd, fname_start="half-", seed=42)

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
    parser.add_argument("--split_letter", type=str, required=True, help="A or B")

    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_arguments()
    # main(args.sample_s, args.split_letter)
  main(1335,"A")
