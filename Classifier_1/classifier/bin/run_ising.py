import sys
sys.path.append("../")
from src.classify_pairwise import Ising_Classifier


if __name__ == "__main__":
    n_categories = 10  # Number of categories to be classified
    n_variables = 121  # Number of variables in the dataset
    mcm_filename_format = "train-images-unlabeled-{}_comms.dat"
    data_filename_format = "train-images-unlabeled-{}.dat"
    data_path = "../INPUT/data/"
    communities_path = "../INPUT/MCMs/"
    output_path = "../OUTPUT/"
    clf = Ising_Classifier(n_categories,n_variables,mcm_filename_format,data_filename_format, data_path,communities_path)

