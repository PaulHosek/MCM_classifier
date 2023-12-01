import numpy as np
import os
import platform

class MCM_Classifier:
    """
    The MCM-classifier

    Args:
        - n_categories (int): The number of categories in the dataset
        - n_variables (int): The number of variables in the dataset
    """

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
