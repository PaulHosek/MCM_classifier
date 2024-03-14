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
    def __init__(self, sample_size, INPUT_dir, fname, all_data_path) -> None:
        self.sample_size = sample_size
        self.INPUT_dir = INPUT_dir
        self.fname = fname
        self.cat_dir = os.path.join(INPUT_dir,fname)
        self.fname_sep_path = os.path.join(self.cat_dir, fname+"_sep")
        self.all_data_dir = all_data_path
        self.is_setup = False

    def setup(self, seed):
        """
        Set up for model fitting: Subsample, convert to spaced format,
          generate .p file, check for no deletion of spins.

        :param seed: The seed value for random number generation.
        :type seed: int
        """

        self.subsample_cat_ace(seed=seed)
        self.convert_to_spaced()
        ACEtools.WriteCMSA("binary", self.fname_sep_path+".dat")
        self.no_deletions()
        self.is_setup = True
    
    def fit(self, method, path_to_exe: str, args: list = []):
        """
        Fit the pairwise model using either ace or qls.

        :param method: The method to use for fitting the model. Valid options are "ace" or "qls".
        :type method: str
        :param path_to_exe: The path to the executable file.
        :type path_to_exe: str
        :param args: Additional arguments to pass to the executable.
        :type args: list
        """
        assert self.is_setup, "Setup has not been completed. Please call the setup() method before fitting the model."

        if method == "ace":
            cml_args = self.build_ace_args(path_to_exe, p_dir=self.cat_dir, p_fname=self.fname+"_sep"+"-output", args=args)
        elif method == "qls":
            cml_args = self.build_qls_args(path_to_exe, p_dir=self.cat_dir, p_fname=self.fname+"_sep"+"-output", args=args)
        else:
            raise ValueError("Invalid method. Please choose either 'ace' or 'qls'.")
        self.call_exec(cml_args)

        f = open(os.devnull, "w")
        p = self.call_exec(cml_args)
        status = p.wait()
        f.close()
    



    def no_deletions(self):
        """
        Verifies that no variables were removed post ACEtools execution.

        This function ensures that no variables, due to never switching to the other state, have been eliminated.
        Such removals can disrupt the sequence in subsequent parts of the program and should be prevented by the subsampler.

        Raises:
            ValueError: An error is raised with the names of the removed variables, if any were deleted.
        """
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
                print(filename)
                if filename.endswith(".dat"):
                    path = os.path.join(root, filename)
                    file = np.genfromtxt(path, dtype=int, delimiter=1)
                    np.savetxt(path[:-4] + "_sep" + ".dat", file, fmt="%d", delimiter=" ")

    def call_exec(self, args: tuple):
        """Call executable.
        Note, this is a sequential implementation. Only one pairwise model is fitted at the time.

        :param args: tuple of what string elements that make up the command line argument
        :type args: tuple
        :raises KeyboardInterrupt: Raised when ACE is interrupted
        :raises SystemExit: Raised when ACE process exits unexpectedly
        :return: The subprocess.Popen object representing the ACE process
        :rtype: subprocess.Popen
        """
        f = open(os.devnull, "w")
        print(args)
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Process interrupted")
        except SystemExit:
            raise SystemExit("Process process exited unexpectedly")
        
        stat = p.wait()
        if stat == 0:
            print(f"\N{check mark} Process done.")
        f.close()



    def build_ace_args(self, path_to_exe:str, p_dir:str ,p_fname:str , args:list, auto_l2=True):
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
        cm_args = [path_to_exe, "-d",p_dir, "-i",p_fname,"-o", p_fname+"-out", "-b", str(self.sample_size)]
        if auto_l2:
            cm_args.append("-ga")
        cm_args.extend(args)

        return tuple(cm_args)

    def build_qls_args(self, path_to_exe:str, p_dir:str, p_fname:str, sample_size:int, args:list, auto_l2=True):
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
        cm_args.extend(args)
        return tuple(cm_args)

    def subsample_cat_ace(self,seed):
        """Clear input_data_path and fill it, from the all_data_path, with a single folder of the category we are interested in.
        
        :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

        """
        rng = np.random.default_rng(seed)

        self.clear_cat(self.INPUT_dir, self.fname)
        # generate new input data 
        for file in os.listdir(self.all_data_dir):
            if file.split(".")[0] == self.fname:
                inp = np.loadtxt(os.path.join(self.all_data_dir,file), dtype="str")
                subfolder_name = file.split(".")[0]  
                subfolder_path = os.path.join(self.INPUT_dir, subfolder_name) 

                os.makedirs(subfolder_path, exist_ok=True) 
                arr = rng.choice(inp, self.sample_size, replace=False)
                arr = np.append(arr, ["0"*121, "1"*121])
                np.savetxt(os.path.join(subfolder_path, file), arr, fmt="%s")

    @staticmethod
    def clear_cat(path, dir_name):
        """In path, delete all folders that match dir_name."""
        for folder in os.listdir(path):
            if folder == dir_name:
                folder_path = os.path.join(path, folder)
                shutil.rmtree(folder_path)









if __name__ == "__main__":
    mod = Pairwise_model(10,"./INPUT/data/","train-images-unlabeled-1", "./INPUT_all/data")
    mod.setup(42)
    mod.fit("ace","./utils/ace")
    # mod.fit("qls", "./utils/")





