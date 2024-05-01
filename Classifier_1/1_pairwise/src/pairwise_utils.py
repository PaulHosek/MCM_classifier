import numpy as np
import sys
sys.path.append("../")
from src.pairwise_fitter import Pairwise_fitter
from src.pairwise_evaluator import Pairwise_evaluator
import os



def get_pw_mod(digit,nspin,fname="train-images-unlabeled-{}",outdir="../OUTPUT_mod/data"):
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
