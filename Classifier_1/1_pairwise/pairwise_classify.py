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


# convert data to spaced format #NOTE UNTESTED!
def convert_to_spaced(INPUT_dir = "INPUT/data"):
    """Converts all files in the INPUT folder and its subfolders into the spaced format."""
    for root, dirs, files in os.walk(INPUT_dir):
        for filename in files:
            if filename.endswith(".dat"):
                path = os.path.join(root, filename)
                file = np.genfromtxt(path, dtype=int, delimiter=1)
                np.savetxt(path[:-4] + "_sep" + ".dat", file, fmt="%d", delimiter=" ")





# call ACEtool code on it to generate .p file
def call_ace(ace_args: tuple):
    """Call Adaptive cluster expansion (ACE) code.
    Note, this is a sequential implementation. Only one pairwise model is fitted at the time.

    :param ace_args: tuple of what string elements that make up the command line argument
    :type ace_args: tuple
    :raises KeyboardInterrupt: Raised when ACE is interrupted
    :raises SystemExit: Raised when ACE process exits unexpectedly
    :return: The subprocess.Popen object representing the ACE process
    :rtype: subprocess.Popen
    """
    with open(os.devnull, "w") as f:
        try:
            p = subprocess.Popen(ace_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("ACE interrupted")
        except SystemExit:
            raise SystemExit("ACE process exited unexpectedly")
        stat = p.wait
    if stat == 0:
        print(f"\N{check mark} ACE done.")
    return p


# test that there were no variables deleted -> check that specific line (should never happen since we always add the all 0s and all 1s but better have safeguard)



# call ACE fitting on it with right flags
# call MC learning QLS algo on it

# at end should have 10 .p files -> one for each category

if __name__ == "__main__":
    helpers.subsample_data_ace(10, all_data_path="./INPUT_all/data", input_data_path="./INPUT/data")

    convert_to_spaced()



    # path = "INPUT/data/train-images-unlabeled-0/train-images-unlabeled-0.dat"
    # res = ACEtools.WriteCMSA("binary",path)

