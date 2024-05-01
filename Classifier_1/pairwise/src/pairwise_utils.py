import numpy as np
import sys
sys.path.append("../")
from pairwise.src.pairwise_fitter import Pairwise_fitter
from pairwise.src.pairwise_evaluator import Pairwise_evaluator
import os


def fit_digit(digit, seed,nsamples,fname = "train-images-unlabeled-{}",\
                inalldir_rel="../../INPUT_all/data/traindata",outdir_rel="../../OUTPUT_mod/data",exe_rel="../../ace_utils/ace"):
    """Fit a pairwise model to the digit until convergence.

    :param digit: The mnist digit (0-9) to fit the model on.
    :type digit: int
    :param seed: The seed value for random number generation.
    :type seed: int
    :param nsamples: The number of samples to use for fitting the model.
    :type nsamples: int
    :param fname: The name of the unlabeled image file, defaults to "train-images-unlabeled-{}".
    :type fname: str, optional
    :param inalldir_rel: The relative path to the input data directory, defaults to "../../INPUT_all/data/traindata".
    :type inalldir_rel: str, optional
    :param outdir_rel: The relative path to the output data directory, defaults to "../../OUTPUT_mod/data".
    :type outdir_rel: str, optional
    :param exe_rel: The relative path to the ACE utility, defaults to "../../ace_utils/ace".
    :type exe_rel: str, optional
    """
    mod = Pairwise_fitter(nsamples,inalldir_rel,fname.format(digit), outdir_rel)
    mod.setup(seed,input_spaced=False)
    mod.fit("ace",exe_rel)

def get_pw_mod(digit,nspin,outdir,fname="train-images-unlabeled-{}",):
    """Get the pairwise model that was fitted on a digit.

    :param digit: mnist digit 0-9. only needed to load the right
    :type digit: int
    :param nspin: number of spins in the system e.g., (121)
    :type nspin: int
    :param fname: name of the .j file directory and file (without _sep-output-out.j), defaults to "train-images-unlabeled-{}"
    :type fname: str, optional
    :param outdir: directory where to fname folder is, defaults to "../OUTPUT_mod/data"
    :type outdir: str, optional
    :return: pairwise model
    :rtype: instance of class Pairwise_evaluator
    """
    fname = fname.format(digit)
    jpath = os.path.join(outdir,fname,fname+"_sep-output-out.j")
    mod = Pairwise_evaluator(jpath, nspin)
    mod.load_ising_paramters()
    return mod

# nspin = 121
# digits = [0,1]
# mods = [get_mod_digit(i,nspin) for i in digits]


