import os
import sys

sys.path.append("../")
import src.paper_utils as utils
import argparse



def main(sample_s):
    
  seed = 1
  n_categories = 10 
  n_variables = 121  
  mcm_filename_format = "full-images-unlabeled-{}_bootstrap_comms.dat"
  data_filename_format = "full-images-unlabeled-{}.dat"
  data_path = "../data/INPUT/data/"
  communities_path = "../data/OUTPUT/mcm/comms/"

  clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path] 
  adp = "../data/INPUT_all/data/combined_data"
  rssd = "../data/OUTPUT/mcm/sample_sizes_fromfull"
  utils.evaluate_subsample(sample_s, clf_args,all_data_path=adp, result_sample_sizes_dir=rssd, fname_start="full-", seed=seed, seed_plus=True)

##########3
  # full sample 6315: train + test
  

def parse_arguments():
    """Parses command-line arguments and returns a dictionary containing them.

  Returns:
    A dictionary containing parsed arguments with keys matching the flag names.
  """
    parser = argparse.ArgumentParser(description="run full sampling regime")
    parser.add_argument("--sample_s", type=int, required=True,
                        help="Sample size.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.sample_s)





