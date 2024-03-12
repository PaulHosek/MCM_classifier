import numpy as np
import sys
sys.path.append("../")
import utils.ACEtools as ACEtools
import helpers
import os
import subprocess


# want class that only has 1 digit 

# need
# for each class


# convert data to spaced format #NOTE UNTESTED!
def convert_to_spaced(INPUT_dir = "INPUT/data"):
    """Converts all files in the INPUT folder into the spaced format."""
    for filename in os.listdir(INPUT_dir):
        if filename.endswith(".txt"):
            path = os.path.join(INPUT_dir, filename)
            file = np.genfromtxt(path, dtype=int, delimiter=1)
            np.savetxt(path + "_sep", file, fmt="%d", delimiter=" ")




# call ACEtool code on it to generate .p file
def call_ace(ace_args: tuple):
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



