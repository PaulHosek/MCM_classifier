import subprocess

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
        P = [] # Probability distributions of Icc configurations for each category
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
                # for each possible configuration, fill in the observed probabilities
                p_icc[u] = c/np.sum(c)
                pk.append(p_icc)
            P.append(pk)

        self.__P = P
        self.__MCM = MCM




    def fit(self, data_path: str = "INPUT/data", greedy: bool = False, max_iter: int = 10_000,
             max_no_improvement: int = 10_000, n_samples: int = 0):

        ##### Generate bootstrapped samples from the data in the data folder ####
        # since in the default setting n_samples is 0, no new samples will be generated
        folder = os.fsencode(data_path)

        sorted_folder = sorted(os.listdir(folder))
        processes = []
        saa_args_list = []

        for file in sorted_folder:
            filename = os.fsdecode(file)
            if not filename.endswith("_bootstrap.dat") and filename.endswith(".dat"):
                # Remove the .dat extension
                filename = filename[:-4]
                if (n_samples != 0):
                    # create new folder for bootstrap samples
                    # if no _bootstrap in filename, add it
                    if "_bootstrap" not in filename:
                        bootstrap_name = filename + "_bootstrap"
                    else:
                        bootstrap_name = filename
                    # os.makedirs("INPUT/data/bootstrap/", exist_ok=True)
                    clf_utils_pp.generate_bootstrap_samples(clf_utils_pp.load_data("INPUT/data/" + filename + ".dat"), bootstrap_name, n_samples)
                    filename = bootstrap_name
                else:
                    if "_bootstrap" not in filename:
                        bootstrap_name = filename + "_bootstrap"
                    else:
                        bootstrap_name = filename
                    clf_utils_pp.generate_bootstrap_samples(clf_utils_pp.load_data("INPUT/data/" + filename + ".dat"), bootstrap_name, len(load_data("INPUT/data/" + filename + ".dat")))
                    filename = bootstrap_name

            #################################

            file = "mcm_classifier/input/data" + filename

            saa_args = self.__construct_args(file,greedy, max_iter, max_no_improvement)
            saa_args_list.append(saa_args)

            for saa_args in saa_args_list:
                f = open(os.devnull, "w")
                p = run_saa(saa_args)
                processes.append((p,f))

            for p,f in processes:
                status = p.wait()
                f.close()
                if status == 0:
                    print(f"\N{check mark} SAA for {saa_args_list[processes.index((p, f))][3].split('/')[-1]} finished successfully")

            print("\N{check mark} Done!")

            self.__construct_P()
    def __construct_args(self, filename: str, greedy:,bool, max_iter: int, max_no_improvement: int) -> tuple:
        """
        Build the argument tuple to pass to the simulated annealing algorithm based on the settings we want.

        :param filename:
        :param greedy:
        :param bool:
        :param max_iter:
        :param max_no_improvement:
        :return:
        """
        operating_system = platform.system()
        g = "-g" if greedy else ""
        sa_file = "../MinCompSpin_SimulatedAnnealing/bin/saa.exe" if operating_system == "Windows" else "../MinCompSpin_SimulatedAnnealing/bin/saa.out"
        saa_args = [sa_file, str(self.n_variables),"-i", filename, g, "--max", str(max_iter), "--stop", str(max_no_improvement)]
        saa_args = tuple(filter(None, saa_args))
        return saa_args

    def run_saa(saa_args: tuple):
        """Runs the MinCompSpin_SimulatedAnnealing algorithm

        Args:
            saa_args (tuple): The arguments for the algorithm

        Returns:
            int: The return code of the algorithm
        """
        try:
            p = subprocess.Popen(saa_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("MinCompSpin_SimulatedAnnealing interrupted")
        except SystemExit:
            raise SystemExit("MinCompSpin_SimulatedAnnealing failed")
        except Exception:
            raise Exception("MinCompSpin_SimulatedAnnealing failed")

        return p

    def classify(self, state: np.ndarray = np.array([])) -> tuple:
        # p is list of
        p = self.__get_probs(state)
        pred = np.argmax(p)

        return pred, p

    def __get_probs(self,state: np.ndarray) -> list:
        """Give list of MCM probabilities for some state."""
        all_probs = []

        for i in range(self.n_categories):
            prob = self.__prob_MCM(state,i)
            all_probs.append(prob)

    def __prob_MCM(self, state: np.ndarray, cat_index: int) -> float:
        """
        P(state|MCM_k)
        Probability of this state, given a specific MCM.
        :param state:
        :param cat_index:
        :return:
        """
        prob = 1
        MCM = self.__MCM[cat_index] # MCM is list of p_icc
        P = self.__P[cat_index]

        # Calculate the product of the individual probabilities of the configuration ICCs in the MCM
        for j, icc in enumerate(MCM):

            # P(state| icc_j)
            # Extrac the state
            # p_icc is an array of probabilities for the state
            # of the binary version of the index e.g., p_icc = [0,0.22,0.12,0...,0.01]
            p_icc = P[j]

            # get the index of that stat in the p_icc array,
            # to extract the probability of the state given this icc

            idx = [i for i in range(self.n_variables) if icc[i] == "1"]
            sm = int("".join([str(s) for s in state[idx]]), base=2)

            # Here happen the 0 probabilities, if the state has never been seen before,
            # then since we initialize the p_icc array with 0s we get 0 for p_icc[sm]
            # this leads the probability of the MCM to to be 0 in total
            # log does not help since then we have -inf = P[log(0)]
            prob *= p_icc[sm]
        return prob




