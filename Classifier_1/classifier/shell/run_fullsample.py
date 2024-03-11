import sys
sys.path.append("../../")
import src.paper_utils as utils
# full sample 6315: train + test
x = utils.evaluate_subsample
# n_categories = 10
# n_variables = 121
# mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
# data_filename_format = "full-images-unlabeled-{}.dat"
# data_path = "../../INPUT/data/"
# communities_path = "../../OUTPUT/comms/"
# output_path = "../../OUTPUT/"
#
# clf_args = [n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path] # data path here for subsampling
#
# full_data_path="../INPUT_all/combined_data"
#
# # sample_sizes = [10]+list(range(0,6315,100))[1:]+[6315]
#
# for sample_s in sample_sizes[55:]:
#
#     for i in range(10):
#         utils.evaluate_subsample(sample_s, clf_args, all_data_path="../../INPUT_all/data/combined_data", result_sample_sizes_dir="../../OUTPUT/sample_sizes_fromfull", fname_start="full-")
#
# # for i in range(7):
# #     utils.evaluate_subsample(5400, clf_args,all_data_path="../INPUT_all/data/combined_data", result_sample_sizes_dir="../OUTPUT/sample_sizes_fromfull", fname_start="full-")
#
