import numpy as np
import os
import platform
import clf_utils_pp

class MCM_Classifier:
    """
    The MCM-classifier

    Args:
        - n_categories (int): The number of categories in the dataset
        - n_variables (int): The number of variables in the dataset
    """

    # Private
    def __init__(self, n_categories: int, n_variables: int, mcm_filename_format: str, data_filename_format: str):
        self.n_categories = n_categories
        self.n_variables = n_variables
        self.__mcm_filename_format = mcm_filename_format
        self.__data_filename_format = data_filename_format

        self.__P = []
        self.__MCM = []
        self.predicted_classes = None
        self.probs = None
        self.stats = None

    # Public
    def init(self):
        self.__construct_P()

    def __construct_P(self) -> tuple:
        """
        Construct Probability distribution over all MCMs/ categories/ digits
        :return:
        """
        P = [] # Probability distributions for each category
        MCM, data = clf_utils_pp.load_all_data_and_mcms(range(self.n_categories),
                                                        self.__mcm_filename_format,
                                                        self.__data_filename_format)

        # MCM is list of strings, where each string is a row of the image
        for mcm in MCM:
            # mcm is binary array of len 128

            pk = []  # Probability distribution of one category
            for icc in mcm:

                # get indices of the current ICC,
                # if diagonal initialization: list of 1 element e.g., [113]
                idx = [i for i in range(self.n_variables) if icc[i] == "1"]
                rank = len(idx)

                # p_icc = array of each possible configuration of the ICC we attribute a probability
                p_icc = np.zeros(2**rank)
                # icc_data = now we look at the data and extract the values for each image for the region of the ICC
                # since here the ICC is rank 1 we get a list of arrays with the 1 element (e.g., binary value at idx 113)
                # if we had more we would get configurations e.g.,for an icc or rank 3:  [[1,0,0], [1,0,0], [0,0,0], [0,0,1]...]
                icc_data = data[:,idx]

                # CONVERT TO INTEGER
                # an individual configuration is a list in icc_data
                # now, we convert this into an integer. e.g., [[0,1,1,1],[0,0,0,1],...] -> [7,1,...]
                icc_strings = []
                for state in icc_data:
                    binary_string = "".join(str(bit) for bit in state)
                    icc_value = int(binary_string, 2)
                    icc_strings.append(icc_value)

                # now each observed state/configuration has been identified with an integer
                # get counts of each configuration [values] [counts] e.g., [0,1,6] [422,1,2]
                u,c = np.unique(icc_strings, return_counts=True)
                p_icc[u] = c/np.sum(c)
                pk.append(p_icc)
            P.append(pk)

        self.__P = P
        self.__MCM = MCM




    def fit(self, data_path: str = "INPUT/data", greedy: bool = False, max_iter: int = 10_000,
             max_no_improvement: int = 10_000, n_samples: int = 0):

