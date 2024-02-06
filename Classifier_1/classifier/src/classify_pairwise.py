from re import S
import numpy as np
import pandas as pd
import subprocess
import os
import platform
import multiprocessing as mp
from sklearn.metrics import jaccard_score, mutual_info_score
from scipy.optimize import line_search

# from Classifier_1.classifier.src.loaders import load_data_toint

# MCM_classifier helper imports
import src.loaders as loaders
import src.helpers as helpers
from src.ising_model import ising_model

class Ising_Classifier:
    def __init__(self, n_categories: int, n_variables: int, mcm_filename_format: str, data_filename_format: str, data_path: str, comms_path: str) -> None:
        """
        The Ising Spin Glass Classifier, limited to pairwise interactions.

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
        self.__P, self.__ISING = ([], [])
        self.predicted_classes = None
        self.probs = None
        self.stats = None
        self.data_path = os.path.join(data_path, "")
        self.comms_path = os.path.join(comms_path, "")

        # self.data = None
        # self.P_models = np.zeros((2**self.n_variables,self.n_categories))
        # self.P_data = np.zeros((2**self.n_variables,self.n_categories))
        # self.spin_op = [i for i in range(2**self.n_variables) if bin(i).count('1') <= 2 and i > 0]
        # self.param = np.random.rand(len(self.spin_op),self.n_categories) # random initialization of model parameters

        self.models = [None]*self.n_categories


        # self.load_data_ising()
        self.setup_models()

    def setup_models(self):
        for model_idx in range(self.n_categories):
            self.models[model_idx] = ising_model(os.path.join(self.data_path, self.__data_filename_format.format(model_idx)))






    def load_data_ising(self)->None:
        """Read all the data for all categories."""

        first_set = loaders.load_data_toint(os.path.join(self.data_path, self.__data_filename_format.format(0)))
        self.data = np.zeros((len(first_set),self.n_categories)) 

        self.data[:,0] = first_set
        for i in range(self.n_categories-1):
            self.data[:,i+1] = loaders.load_data_toint(os.path.join(self.__data_filename_format.format(i+1)))

    def __get_n_categories(self):
        return self.n_categories
    def __get_n_variables(self):
        return self.n_variables
    def __get_P_model(self,category):
        return self.P_models[:,category]
    def __get_P_data(self,category):
        return self.P_data[:,category]
    def __get_spin_op(self):
        return self.spin_op
    def __get_model_parameters(self,category):
        return self.param[:,category]
    def __get_class_data(self,category):
        if self.data == None:
            raise ValueError("Data not loaded.")
        return self.data[:,category]
        







































    # def setup_models(self):
    #     if self.data is None:
    #         raise ValueError("Data not loaded.")
    #     # build empirical distribution for each category
    #     for k in range(self.n_categories):
    #         emp_distr = np.empty(2**self.n_variables)
    #         for state in range(2**self.n_variables):
    #             emp_distr[state] = np.count_nonzero(self.data[:,k] == state)
    #         self.P_data[:,k] = emp_distr / len(self.data[:,k]) 


    # def fit_single(self, model_idx, n_iter =500):

    #     for _ in range(n_iter):
    #         param = self.param[model_idx]
    #         s = -self.calc_jacobian()
    #         alpha = scipy.optimize.line_search(self.f_x, self.grad_f, param, s)[0]

    #         self.set_param(model_idx, param + alpha *s)

    # def set_param(self, model_idx, new_param):
    #     self.param[model_idx] = new_param
    #     self.calc_model_distr()

    # def calc_model_distr(self, model_idx):
    #     """Calculate state distribution of the model with current parameters."""
    #     g = np.zeros(2**self.n_variables)
    #     g[self.spin_op] = self.param
    #     energy = self.fwht(g)
    #     self.P_models[:,model_idx] = 



    




        

        

        












# # FROM ARONDC60
# class Ising_model:

#     def __init__(self,classifier, model_idx):
#         """
#         Initialize an Ising model object.

#         Parameters
#         ----------
#         file : str
#             path to the file containing the data
#         """
#         # List containing the model distribution
#         self.classifier = classifier
#         self.model_idx = model_idx
#         self.calc_emp_distr()
#         self.data = self.classifier.__get_P_data(self.model_idx)

    
#     def set_param(self, param):
#         """
#         Set the values for the model paramaters.

#         Parameters
#         ----------
#         param : list 
#             new values for the model parameters
#         """
#         self.param = param
#         # Recalculate the model distribution with new parameters
#         self.calc_model_distr()
    
#     def calc_emp_distr(self):
#         """
#         Calculate the empirical distribution
#         """
#         emp_distr = np.empty(2**self.classifier.__get_n_variables())
#         for state in range(2**self.classifier.__get_n_variables()):
#             emp_distr[state] = np.count_nonzero(data == state)
#         return emp_distr / len(data)


#     def calc_model_distr(self):
#         """Calculate the model distribution for the current model parameters."""
#         g = np.zeros(2**self.classifier.__get_n_variables())
#         g[self.spin_op] = self.param

#         energy = fwht(g)
#         model_distr = np.exp(energy)
#         self.model_distr = model_distr / np.sum(model_distr)
    
#     def calc_exp_model(self):
#         """
#         Calculate the expected value for the spin operators given the current model parameters.

#         Returns
#         -------
#         <phi_mu> : array
#             expected value for every spinoperator
#         """
#         exp_model = utils.fwht(self.model_distr)
#         return exp_model[self.spin_op]

#     def calc_KL_div(self):
#         """
#         Calculate the Kullback-Leibler divergence between the empirical distribution and the model distribution.

#         Returns
#         ------
#         kl_div : float
#             KL divergence
#         """
#         div = self.emp_distr / self.model_distr
#         log = np.log(div, out=np.zeros_like(div), where=div!=0)

#         kl_div = self.emp_distr @ log

#         return kl_div

#     def calc_jacobian(self):
#         """
#         Calculate the jacobian (derivative of the negative loglikelihood with respect to the model parameters.)

#         Returns
#         -------
#         jacobian : array
#             Jacobian given the current model parameters
#         """
#         jacobian = fwht(self.model_distr - self.emp_distr)
#         return jacobian[self.spin_op]
    
#     def f_x(self, param):
#         """
#         Function to minimize ((KL divergence) in the steepest descent algorithm.

#         Parameters
#         ----------
#         param : array
#             new model parameters
        
#         Returns
#         -------
#         kl_div : float
#             KL divergence
#         """
#         # Set new parameters (+ recalculate model distribution)
#         self.set_param(param)
#         # Calculate new KL divergence with the empirical distribution
#         return self.calc_KL_div()
    
#     def grad_f(self, param):
#         """
#         Gradient of the function to minimize in the steepest descent algorithm.

#         Parameters
#         ----------
#         param : array
#             new model parameters
        
#         Returns
#         -------
#         jacobian : array
#             Jacobian for the given model parameters
#         """
#         # Set new parameters (+ recalculate model distribution)
#         self.set_param(param)
#         # Calculate the new gradient
#         return self.calc_jacobian()

#     def fit_param(self, n_iter = 500):
#         """
#         Find model parameters using the steepest descent algorithm.

#         Parameters
#         ----------
#         n_iter : int
#             Maximum number of iterations in the algorithm
#         """
#         for _ in range(n_iter):

#             param = self.param
#             s = - self.calc_jacobian()
#             # perform a linesearch
#             result = scipy.optimize.line_search(self.f_x, self.grad_f, param, s)
#             alpha = result[0]

#             # Update the model parameters
#             self.set_param(param + alpha * s)

#             # Check convergence
#             if np.allclose(self.param, param):
#                 break



