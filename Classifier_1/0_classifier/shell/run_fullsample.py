import sys
sys.path.append("../")
import src.paper_utils as utils


def main(sample_s):
# full sample 6315: train + test
    n_categories = 10
    n_variables = 121
    mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
    data_filename_format = "full-images-unlabeled-{}.dat"
    data_path = "../../INPUT/data/"
    communities_path = "../../OUTPUT/comms/"

    clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path] # data path here for subsampling


    # sample_sizes = [10]+list(range(0,6315,100))[1:]+[6315]


    for s in range(10):
        seed = s + sample_s
        utils.evaluate_subsample(sample_s, clf_args, all_data_path="../../INPUT_all/data/combined_data", result_sample_sizes_dir="../../OUTPUT/sample_sizes_fromfull", fname_start="full-", seed=seed)


if __name__ == "__main__":
    sample_s= sys.argv[1]
    main(sample_s)