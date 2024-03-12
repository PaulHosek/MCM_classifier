




# need
# for each class

# convert data to spaced format
# call ACEtool code on it to generate .p file
# test that there were no variables deleted -> check that specific line (should never happen since we always add the all 0s and all 1s but better have safeguard)
# call ACE fitting on it with right flags
# call MC learning QLS algo on it

# at end should have 10 .p files -> one for each category

import numpy as np
import sys
sys.path.append("../")
import ACE.scripts.ACEtools as ACEtools




