from matplotlib.backend_bases import key_press_handler
import numpy as np
import pandas as pd
import subprocess
import os
import platform
import multiprocessing as mp
from sklearn.metrics import mutual_info_score

# MCM_classifier helper imports
from .loaders import load_data, load_mcm
from .helpers import generate_bootstrap_samples, print_box


class MCM_Classifier:
    """
    The MCM-classifier

    Args:
        - n_categories (int): The number of categories in the dataset
        - n_variables (int): The number of variables in the dataset
    """

    def __init__(self, n_categories: int, n_variables: int, mcm_filename_format: str, data_filename_format: str, data_path: str, comms_path: str) -> None:
        """
        The MCM-classifier.

        Args:
            - n_categories (int): The number of categories in the dataset
            - n_variables (int): The number of variables in the dataset
            - mcm_filename_format (str): The format of the MCM filenames
            - data_filename_format (str): The format of the data filenames
        """
        self.n_categories = n_categories
        self.n_variables = n_variables
        self.__mcm_filename_format = mcm_filename_format
        self.__data_filename_format = data_filename_format

        # Construct probability distributions and MCMs for each category
        self.__P, self.__MCM = ([], [])
        self.__Counts = []
        self.predicted_classes = None
        self.probs = None
        self.stats = None
        self.data_path = os.path.join(data_path, "")
        self.comms_path = os.path.join(comms_path, "")

        
    # ----- Public methods -----
    def init(self):
        """
        Initializes the classifier if the MCMs have already been selected.
        """
        self.__construct_P()
    
    def get_MCMs(self):
        return self.__MCM
    
    def get_P(self):
        return self.__P
    
    def get_Counts(self):
        return self.__Counts


    def fit(self, 
            greedy: bool = False, max_iter: int = 100000, max_no_improvement: int = 10000, n_samples: int = 0) -> None:
        """
        Fit the classifier using the data given in the data_path folder.
        It uses the MinCompSpin_SimulatedAnnealing algorithm to find the MCMs.

        Args:
            - greedy (bool): Whether to use the greedy algorithm after SA
            - max_iter (int): Maximum number of iterations for the SA algorithm
            - max_no_improvement (int): Maximum number of iterations without improvement for the SA algorithm
            - n_samples (int): The number of samples to be used from the data folder. If 0, all samples are used.
        """
        # if not self.__validate_input_data():
        #     raise ValueError("Input data folder file count does not match number of categories")
        # Loop over each file in the data folder
        fit_args = (greedy, max_iter, max_no_improvement,n_samples)
        saa_args_list = self.__get_saa_args_and_bootstrap(fit_args)
        # Run the MinCompSpin_SimulatedAnnealing algorithm on different processes for each file
        # TODO maybe not so good to run a process per category, do not know the nr of categories
        print_box("Running MinCompSpin_SimulatedAnnealing...")
        processes = []
        for saa_args in saa_args_list:
            f = open(os.devnull, 'w')
            p = self.run_saa(saa_args)
            processes.append((p, f))
        # Wait for all processes to finish
        for p, f in processes:
            status = p.wait()
            f.close()
            if status == 0:
                print(f"\N{check mark} SAA for {saa_args_list[processes.index((p, f))][3].split('/')[-1]} finished successfully")
        print("\N{check mark} Done!")
        
        # Construct probability distributions and MCMs for each category
        self.__construct_P()
   
    def classify(self, state: np.ndarray = np.array([])) -> tuple:
        """
        Classify a single state using the MCM-based classifier.
        
        Args:
            state (np.ndarray): The state to be classified
        """
        # ----- Calculate probability of sample belonging to each category -----
        probs = self.__get_probs(state)
        print(probs)
        predicted_class = np.argmax(probs)
         
        return predicted_class, probs

    def predict(self, data: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Evaluates the performance of the MCM-based classifier.

        Args:
            data (np.ndarray): The data to be classified
            labels (np.ndarray): The labels of the data
            
        Returns:
            tuple: The predicted classes (for each state) and the probabilities for each category (for each state)
        """
        print_box("Evaluating classifier...")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if len(self.__P) == 0 or len(self.__MCM) == 0:
            raise ValueError("Classifier not initialized yet. If you have already selected MCMs, try running the init method first. If not, try running the fit method first.")

        # ----- Calculate probability of sample belonging to each category -----
        print_box("1. Calculating state probabilities...")
        # if method == "probs":
        probs = np.array([self.__get_probs(state) for state in data])
        # elif method ==  "kl":
        #     probs = np.array([self.__get_MI(state) for state in data])
        # ----- Predict classes -----   
        predicted_classes = np.argmax(probs, axis=1)
        
        # If all of the probabilities for a state are 0, predict the class as -1
        predicted_classes[np.all(probs == 0, axis=1)] = -1
        # Print how many states were not classified
        print("Number of datapoints for which the classifier didn't have any probability for any category: {}".format(len(predicted_classes[predicted_classes == -1])))
        # ----- Calculate accuracy -----
        print_box("2. Calculating accuracy...")

        correct_predictions = predicted_classes == labels
        accuracy = np.sum(correct_predictions) / len(labels)

        # ----- Save stats -----
        print_box("3. Saving stats...")

        self.predicted_classes = predicted_classes
        self.probs = probs
        self.stats = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "confusion_matrix": self.__get_confusion_matrix(labels),
        }

        print("\N{white heavy check mark} Done!")

        return predicted_classes, probs

    #  ---------------------------- Probability calculation methods ----------------------------

    def __construct_P(self, estimator="add_smooth",alp=1) -> tuple:
        """
        Construct probability distributions for each category.
        This function should only be run once during initalization of the classifier.

        Code provided by https://github.com/ebokai

        Args:
            P (list): List of the probability distributions for each category
            MCM (list): List of the MCMs for each category
            n_categories (int): Number of categories in the dataset
            n_variables (int): Number of variables in the dataset
        """
        MCM = []
        P = []
        

        print_box("Constructing probability distributions...")

        # if not self.__validate_input_comms():
            # raise ValueError("Input data folder file count does not match number of categories. Did you run the fit method?.")

        # Construct probability distributions for each category
        for k in range(self.n_categories):
            # Add MCM to list
            try:
                # mcm = load_mcm(f"../INPUT/MCMs/{self.__mcm_filename_format.format(k)}")

                mcm = load_mcm(os.path.join(self.comms_path, self.__mcm_filename_format.format(k)))
                MCM.append(mcm)
            except:
                # Throw error if MCM file not found
                raise FileNotFoundError(f"Could not find MCM file for category {k}")

            # Load data
            try:
                # data = load_data(f"../INPUT/data/{self.__data_filename_format.format(k)}")
                data = load_data(os.path.join(self.data_path, self.__data_filename_format.format(k)))
            except:
                # Throw error if data file not found
                raise FileNotFoundError(f"Could not find data file for category {k}")
            pk = []
            p_count_k = []

            for icc in mcm:
                # TODO maybe we later would like a single call to our estimator function
                #  here if we are also using some KL-divergence-based approach that works very differently
                idx = [i for i in range(self.n_variables) if icc[i] == "1"]
                rank = len(idx)

                # p_icc = np.zeros(2**rank)
                # p_icc = np.full(2**rank,fill_value=np.nan) # initialized with 0, lets try nan
                # p_icc = np.full(2**rank,fill_value=1/(2**rank+2)) # laplacian smoothing for 0 observations: (0+1)/(N+2*1)
                p_icc = self.estimator_init(rank,method=estimator,alpha=alp) # TODO this takes a lot of space, maybe we could find a sparse replacement
                p_count = np.zeros(2**rank)
                icc_data = data[:, idx]
                icc_strings = [
                    int("".join([str(s) for s in state]), 2) for state in icc_data
                ]

                u, c = np.unique(icc_strings, return_counts=True)
                p_icc[u] = self.estimator_prob(c,rank,method=estimator,alpha=alp,)
                p_count[u] = c
                # p_icc[u] = c / np.sum(c)
                p_count_k.append(list(p_count))
                pk.append(p_icc)

            P.append(pk)
            self.__Counts.append(p_count_k)

        self.__P = P
        self.__MCM = MCM

        return self.__P, self.__MCM

    @staticmethod
    def estimator_init(rank, method="mle", alpha=1):
        """Iniitalizes array for one icc based on the estimator used."""
        d = rank
        if method == "mle":
            return np.zeros(2**rank)
        elif method == "add_smooth":
            return np.full(2**rank,fill_value=alpha/(d*alpha)) # laplacian smoothing for 0 observations: (0+alpha)/(N+2*alpha)
        else:
            ValueError("Invalid probability estimation method")
    
    @staticmethod
    def estimator_prob(counts,rank, method="mle", alpha=1):
        """
        Used probability estimator function.
        method options are:
        - "mle" = uses pure counts. Maximum Likelihood estimate of the probabilities. Can lead to 0 probabilities.
        - "add_smooth" = additive smoothing, give alpha parameter. 1 = laplacian smoothening/uniform prior, 0.5 = jeffrys prior
            - (x_i + alpha) / (N + alpha*d), where d is 2 because of binary data
        Args:
            method (str, optional): The method used for probability estimation. Defaults to "mle".
            alpha (float, optional): The alpha value for the estimator. Defaults to None.

        Returns:
            probability estimate
        """
        d = rank
        if method == "mle":
            return counts / np.sum(counts)
        elif method == "add_smooth":
            return (counts+alpha)/ (np.sum(counts)+d*alpha) 
        else:
            raise ValueError("Invalid probability estimation method")
        
    def __get_probs(self, state: np.ndarray) -> list:
        """
        Get the probabilities for a single state for each category, in order
        
        Args:
            state (np.ndarray): The state to calculate the probability of
        """
        all_probs = []
        for i in range(self.n_categories):
            prob = self.__prob_MCM(state, i)
            all_probs.append(prob)
        return all_probs

    def __prob_MCM(self, state: np.ndarray, cat_index: int) -> float:
        """
        Calculate the probability of a state given a single MCM.

        Loop through each ICC and calculate the probability of the state
        1. Get the probability distribution restricted to specific ICC
        2. Get the state of the variables in the ICC and convert to binary string
        3. Multiply the probability of the state by the probability of the ICC-state

        Args:
            P (np.ndarray): Probability distributions for one category
            MCM (np.ndarray): MCM for one category
            state (np.ndarray): The state to calculate the probability of
        """

        prob = 1
        MCM = self.__MCM[cat_index]
        P = self.__P[cat_index]

        for j, icc in enumerate(MCM):
            p_icc = P[j]
            idx = [i for i in range(self.n_variables) if icc[i] == "1"]
            ss = state[idx]
            sm = int("".join([str(s) for s in ss]), 2)
            prob *= p_icc[sm]

        return prob

    def __get_mi(self, state: np.ndarray) -> list:
        """
        Get the mutual information for a single state for each category, in order
        
        Args:
            state (np.ndarray): The state to calculate the probability of
        """
        all_mi = []
        for i in range(self.n_categories):
            mi = self.__mi_MCM(state,i)
            all_mi.append(mi)
        return all_mi

    def __mi_MCM(self, state,cat_index):

        MCM = self.__MCM[cat_index]
        P = self.__P[cat_index]
        info = 0
        p_x = 1
        for j, icc in enumerate(MCM):
            p_icc = P[j]
            idx = [i for i in range(self.n_variables) if icc[i] == "1"]
            ss = state[idx]
            sm = int("".join([str(s) for s in ss]), 2)

            info += p_x * p_icc[sm]*np.log(p_x/p_icc[sm]) # this is not defined if p_icc[sm] = 0
        pass



    


    #  ---------------------------- SAA methods ----------------------------

    def __construct_args(self, filename: str, greedy: bool, max_iter: int, max_no_improvement: int) -> tuple:
        """
        Generates the arguments for the MinCompSpin_SimulatedAnnealing algorithm

        Args:
            filename (str): name of the datafile without any path
            greedy (bool): if greedy should be done instead of SAA
            max_iter (int): maximal number of iterations before stopping
            max_no_improvement (int): max nr of iterations without improvement found before stopping

        Returns:
            list: The list with all the arguments, to be used in the subprocess call
        """

        g = "-g" if greedy else ""

        sa_file = "../../MinCompSpin_SimulatedAnnealing/bin/saa.exe" if platform.system() == "Windows" else "../../MinCompSpin_SimulatedAnnealing/bin/saa.out" # TODO use os.path.join here instead
        saa_args = [sa_file,
                    str(self.n_variables),
                    '-i', # this flag indicates starting from single spin basis.
                    filename,
                    g,
                    '--max',
                    str(max_iter),
                    '--stop',
                    str(max_no_improvement)
                    ] 

        # Filter out empty strings
        return tuple(filter(None, saa_args))


    @staticmethod
    def run_saa(saa_args: tuple):
            """Runs the MinCompSpin_SimulatedAnnealing algorithm

            Args:
                saa_args (tuple): The arguments for the algorithm

            Returns:
                int: The return code of the algorithm
                """
            try:
                p = subprocess.Popen(saa_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # ('../bin/saa.exe', str(n), '-i', datafile)
                # print(saa_args)
                # p = subprocess.Popen(saa_args,subprocess.PIPE)
                # print(p)
                # for line in p.stdout:
                #     print(line[:-1].decode('utf-8'))
            except KeyboardInterrupt:
                raise KeyboardInterrupt("MinCompSpin_SimulatedAnnealing interrupted")
            except SystemExit:
                raise SystemExit("MinCompSpin_SimulatedAnnealing failed")
            except Exception:
                raise Exception("MinCompSpin_SimulatedAnnealing failed")

            return p

    def __get_saa_args_and_bootstrap(self, fit_args):
        folder = os.fsencode(self.data_path)
        sorted_folder = sorted(os.listdir(folder))
        saa_args_list = []
        greedy, max_iter, max_no_improvement, n_samples = fit_args
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
                    generate_bootstrap_samples(self.data_path, filename, bootstrap_name, n_samples)
                    filename = bootstrap_name
                else:
                    if "_bootstrap" not in filename:
                        bootstrap_name = filename + "_bootstrap"
                    else:
                        bootstrap_name = filename
                    generate_bootstrap_samples(self.data_path, filename, bootstrap_name, len(load_data(self.data_path + filename + ".dat")))
                    filename = bootstrap_name

                # file = "mcm_classifier/input/data/" + filename
                saa_args = self.__construct_args(filename, greedy, max_iter, max_no_improvement)
                saa_args_list.append(saa_args)
            else:
                continue
        return saa_args_list











# ----------------------------------------------------------------------------------------------------------------------
# less relevant methods
# ----------------------------------------------------------------------------------------------------------------------

    def sample_MCM(self, n_samples: int):
        """
        Samples n_samples from the MCMs randomly

        Args:
            n_samples (int): The number of samples to be generated

        Returns:
            list: A list of the generated samples
        """
        samples = []

        for i in range(n_samples):
            category = np.random.randint(0, self.n_categories)
            samples.append(self.__sample_MCM(category))

        return samples

    def get_classification_report(self, labels: np.ndarray) -> dict:
        """
        Get the classification report for the classifier

        Args:
            labels (np.ndarray): The labels of the data

        Raises:
            ValueError: If the classifier has not been evaluated yet

        Returns:
            dict: The classification report
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        # Get the confusion matrix
        confusion_matrix = self.__get_confusion_matrix(labels)

        correct_predictions = self.predicted_classes == labels
        accuracy = np.sum(correct_predictions) / len(labels)
        # Calculate the precision, recall and f1-score for each category
        precision = np.zeros(self.n_categories)
        recall = np.zeros(self.n_categories)
        f1_score = np.zeros(self.n_categories)

        for i in range(self.n_categories):
            # Calculate precision
            precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])

            # Calculate recall
            recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

            # Calculate f1-score
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        # Calculate the average precision, recall and f1-score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1_score = np.mean(f1_score)

        # Calculate the accuracy
        non_rejected_accuracy = np.sum(correct_predictions) / (len(labels) - np.sum(self.predicted_classes == -1))
        # Include the number of rejected samples in the accuracy calculation
        true_accuracy = accuracy
        rejected = self.predicted_classes == -1
        classification_quality = np.sum(correct_predictions[~rejected]) / (
                    np.sum(correct_predictions[~rejected]) + np.sum(rejected))
        # Construct the classification report
        classification_report = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1_score,
            "true_accuracy": true_accuracy,
            "rejected": len(self.predicted_classes[rejected]),
            "non_rejected_accuracy": non_rejected_accuracy,
            "classification_quality": classification_quality,
        }

        return classification_report

    def save_classification_report(
            self, labels: np.ndarray, path, name: str = "classification_report"
    ) -> None:
        """
        Saves the classification report to a file

        Args:
            name (str): The name of the file
            path (str): The path to the folder where the file should be saved
            labels (np.ndarray): The labels of the data
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        # Get the classification report
        classification_report = self.get_classification_report(labels)
        self.__save_classification_report(name, classification_report, path)


    def __save_classification_report(self, name: str, classification_report: dict, path: str):
        """
        Saves the classification report to a file

        Args:
            name (str): The desired name of the file
            classification_report (dict): The classification report
            path (str): The path to the folder where the file should be saved
        """
        with open(f"{path}/{name}.txt", "w+") as f:
            f.write("Classification report:\n")
            f.write(f"Accuracy: {classification_report['true_accuracy']}\n")
            f.write(f"Average precision: {classification_report['avg_precision']}\n")
            f.write(f"Average recall: {classification_report['avg_recall']}\n")
            f.write(f"Average f1-score: {classification_report['avg_f1_score']}\n")
            f.write("\n")
            f.write("Precision:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['precision'][i]}\n")
            f.write("\n")
            f.write("Recall:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['recall'][i]}\n")
            f.write("\n")
            f.write("F1-score:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['f1_score'][i]}\n")


    def __str__(self) -> str:
        """
        String representation of the classifier
        """
        return f"MCM_Classifier(n_categories={self.n_categories}, n_variables={self.n_variables})"


    def __validate_input_data(self) -> bool:
        """
            Validates the input community folder. Checks if the number of files in the folder
            is equal to the number of categories.
        """
        folder = os.fsencode(self.data_path)
        sorted_folder = sorted(os.listdir(folder))

        n_matching_files = 0
        for file in sorted_folder:
            filename = os.fsdecode(file)
            print(filename)
            print(self.__data_filename_format.format(n_matching_files))
            if filename == self.__data_filename_format.format(n_matching_files):
                n_matching_files += 1

        if n_matching_files == self.n_categories: return True
        return False


    def __validate_input_comms(self) -> bool:
        """
            Validates the input community folder. Checks if the number of files in the folder
            is equal to the number of categories.
        """
        folder = os.fsencode(self.comms_path)
        sorted_folder = sorted(os.listdir(folder))

        n_matching_files = 0
        for i, file in enumerate(sorted_folder):
            filename = os.fsdecode(file)
            if filename == self.__mcm_filename_format.format(i):
                n_matching_files += 1

        if n_matching_files == self.n_categories: return True
        return False

    def __get_confusion_matrix(self, test_labels: np.ndarray):
        """
        Get the confusion matrix for the classifier

        Args:
            test_labels (np.ndarray): The labels of the test data

        Raises:
            ValueError: If the classifier has not been evaluated yet

        Returns:
            np.ndarray: The confusion matrix
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        confusion_matrix = np.zeros((self.n_categories, self.n_categories))
        for i, label in enumerate(test_labels):
            if self.predicted_classes[i] != -1:
                confusion_matrix[label, self.predicted_classes[i]] += 1
            else:
                confusion_matrix[label, label] += 1

        return confusion_matrix

    def __sample_MCM(self, cat_index: int) -> np.ndarray:
        """
        Sample a state from some MCM.

        Args:
            cat_index (int): The category index from which to sample
        """
        # get a sample for each digit

        pk = self.__P[cat_index]  # probability distribution for each digit
        mcm = self.__MCM[cat_index]  # communities for each digit

        sampled_state = np.zeros(self.n_variables)

        for j, icc in enumerate(mcm):
            p_icc = pk[j]  # get the probability distribution restricted to specific ICC
            idx = [
                i for i in range(self.n_variables) if icc[i] == "1"
            ]  # count the number of variables in ICC
            rank = len(idx)
            sm = np.random.choice(np.arange(2 ** rank), 1, p=p_icc)[
                0
            ]  # sample "random" state of ICC
            ss = format(sm, f"0{rank}b")  # convert integer to binary string
            ss = np.array([int(s) for s in ss])  # convert binary string to [0,1] array
            sampled_state[idx] = ss  # fill ICC part of complete state

        return sampled_state
