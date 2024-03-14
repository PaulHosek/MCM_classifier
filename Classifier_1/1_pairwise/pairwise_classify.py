import numpy as np
import sys
sys.path.append("../")
import utils.ACEtools as ACEtools
import helpers
import os
import subprocess
import shutil


# want class that only has 1 digit 

# need
# for each class

class Pairwise_model():
    # fit single pairwise model/ single category
    def __init__(self, sample_size, INPUT_dir, fname) -> None:
        self.sample_size = sample_size
        self.INPUT_dir = INPUT_dir
        self.fname = fname
        self.cat_dir = os.path.join(INPUT_dir,fname)

    def fit():
    # make this a class and one main function to do the steps in order
                            # subsample, convert, ACEtools, nodeletions
         helpers.subsample_data_ace(10, all_data_path="./INPUT_all/data", input_data_path="./INPUT/data")
        convert_to_spaced()

    # path = "INPUT/data/train-images-unlabeled-0/train-images-unlabeled-0_sep.dat"
    # res = ACEtools.WriteCMSA("binary",path)
    # no_deletions()

    pass


# test that there were no variables deleted -> check that specific line (should never happen since we always add the all 0s and all 1s but better have safeguard)
    def no_deletions(self):
        """Check after running ACEtools, if there were variables deleted because they were always 0.
        This should not happen as it messes up the order later. The subsampler should avoid that this happens.
        This is a function to perform this sanity check."""
        for root, dirs, files in os.walk(self.cat_dir):
            for filename in files:
                if filename.endswith(".rep"):
                    path = os.path.join(root, filename)
                    with open(path, 'r') as file:
                        lines = file.readlines()
                        if len(lines) >= 8 and lines[7].strip() != "":
                            raise ValueError("Variables were deleted: \n" + lines[7].strip())



    def convert_to_spaced(self):
        """Converts all files in the current CATEGORY's folder and its subfolders from binary strings to binary integers with spaces:
        e.g., 000111 -> 0 0 0 1 1 1.
        """
        for root, dirs, files in os.walk(self.cat_dir):
            for filename in files:
                if filename.endswith(".dat"):
                    path = os.path.join(root, filename)
                    file = np.genfromtxt(path, dtype=int, delimiter=1)
                    np.savetxt(path[:-4] + "_sep" + ".dat", file, fmt="%d", delimiter=" ")





# call ACEtool code on it to generate .p file
    def call_exec(args: tuple):
        """Call executable.
        Note, this is a sequential implementation. Only one pairwise model is fitted at the time.

        :param args: tuple of what string elements that make up the command line argument
        :type args: tuple
        :raises KeyboardInterrupt: Raised when ACE is interrupted
        :raises SystemExit: Raised when ACE process exits unexpectedly
        :return: The subprocess.Popen object representing the ACE process
        :rtype: subprocess.Popen
        """
        with open(os.devnull, "w") as f:
            try:
                p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt("Process interrupted")
            except SystemExit:
                raise SystemExit("Process process exited unexpectedly")
            stat = p.wait
        if stat == 0:
            print(f"\N{check mark} Process done.")
        return p




    def build_ace_args(path_to_exe:str, p_dir:str ,p_fname:str , sample_size:int, auto_l2=True, *args):
        """Generate the arguments for the ACE algorithm.

        :param path_to_exe: path to the executable file. e.g., ./bin/ace
        :type path_to_exe: str
        :param p_dir: directory of the .p file to do the ace on. e.g., ./INPUT/data/img1
        :type p_dir: str
        :param p_fname: filename of the .p file e.g., img1_sep
        :type p_fname: str
        :param sample_size: sample size the .p size is based on
        :type sample_size: int
        :param auto_l2: if should use automatice 1/sample_size l2 regularisation. Not gauge invariant., defaults to True
        :type auto_l2: bool, optional
        :return: argument tuple to pass to subprocess.Popen
        :rtype: tuple
        """
        cm_args = [path_to_exe, "-d",p_dir, "-i",p_fname,"-o", p_fname+"-out", "-b", str(sample_size)]
        if auto_l2:
            cm_args.append("-ga")
        cm_args.append(args)

        return tuple(cm_args)

    def build_qls_args(path_to_exe:str, p_dir:str, p_fname:str, sample_size:int, auto_l2=True, *args):
        """Generate the arguments for the MC algorithm.

        :param path_to_exe: path to the executable file. e.g., ./bin/ace
        :type path_to_exe: str
        :param p_dir: directory of the .p file to do the ace on. e.g., ./INPUT/data/img1
        :type p_dir: str
        :param p_fname: filename of the .p file e.g., img1_sep
        :type p_fname: str
        :param sample_size: sample size the .p size is based on
        :type sample_size: int
        :param auto_l2: if should use automatice 1/sample_size l2 regularisation. Not gauge invariant., defaults to True
        :type auto_l2: bool, optional
        :return: argument tuple to pass to subprocess.Popen
        :rtype: tuple
        """
     
        cm_args = [path_to_exe, "-d", p_dir, "-c", p_fname,"-w",p_fname,"-i",p_fname+"-out", "-o",p_fname+"-out-fit","-b",str(sample_size)]
        if auto_l2:
            cm_args.append("-ga")
        cm_args.append(args)
        return tuple(cm_args)

# call ACE fitting on it with right flags
# call MC learning QLS algo on it

# at end should have 10 .p files -> one for each category
                    
# make this a class and one main function to do the steps in order
                    # subsample, convert, ACEtools, nodeletions





# def convert_to_spaced(INPUT_dir = "INPUT/data"):
#     """Converts all files in the INPUT folder and its subfolders from binary strings to binary integers with spaces:
#     e.g., 000111 -> 0 0 0 1 1 1.
#     """
#     for root, dirs, files in os.walk(INPUT_dir):
#         for filename in files:
#             if filename.endswith(".dat"):
#                 path = os.path.join(root, filename)
#                 file = np.genfromtxt(path, dtype=int, delimiter=1)
#                 np.savetxt(path[:-4] + "_sep" + ".dat", file, fmt="%d", delimiter=" ")


if __name__ == "__main__":
    # helpers.subsample_data_ace(10, all_data_path="./INPUT_all/data", input_data_path="./INPUT/data")
    # convert_to_spaced()

    # path = "INPUT/data/train-images-unlabeled-0/train-images-unlabeled-0_sep.dat"
    # res = ACEtools.WriteCMSA("binary",path)
    # no_deletions()





