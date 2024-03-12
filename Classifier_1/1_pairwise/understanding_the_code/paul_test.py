import numpy as np
import ACEtools as tools

# add spaces between 0s and 1s
# file = np.genfromtxt(path.format(""),dtype=int,delimiter=1)
# np.savetxt(path.format("_sep"),file, fmt="%d",delimiter=" ")

path = "../../trials/test/img_0/test-images-unlabeled-0{}.dat" # res = tools.WriteCMSA("binary",path.format("_sep"))
path = "../../trials/test/img_0_train/train-images-unlabeled-0{}.dat" # res = tools.WriteCMSA("binary",path.format("_sep"))

path_nspin = "../../trials/test/{}_spin/{}spin.dat"

if __name__ == "__main__":

    # res = tools.WriteCMSA("binary",path_nspin.format(121,121))
    res = tools.WriteCMSA("binary",path.format("_sep"))

    # ../bin/ace -d ../../trials/test/img_0_train -i train-images-unlabeled-0_sep-output -o train-images-unlabled-0_sep-output-out -g2 0.0002 



    # ../bin/ace -d ../../trials/test/4_spin -i 4spin -o 4spin-out -g2 0.01
    # ACE 
    # ../bin/ace -d ../../trials/test/img_0 -i test-images-unlabeled-0_sep-output -o test-images-unlabled-0_sep-output-out -g2 0.0002 
    # from scripts for 2nd step
    # ../bin/qgt -d ../../trials/test/img_0 -c test-images-unlabeled-0_sep-output -i test-images-unlabled-0_sep-output -o test-images-unlabled-0_sep-output-learn -g2 0.0002

    