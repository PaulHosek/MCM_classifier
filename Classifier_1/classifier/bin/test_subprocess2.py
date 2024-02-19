import numpy
import subprocess
import os

#../../MinCompSpin_SimulatedAnnealing/bin/saa.out 121 -i train-images-unlabeled-0_bootstrap -g --max 1000000 --stop  100000
if __name__ == "__main__":
    saa_args = ('../../MinCompSpin_SimulatedAnnealing/bin/saa.out', '121', '-i', 'train-images-unlabeled-0_bootstrap', '-g', '--max', '1000000', '--stop', '100000')

#
    processes = []

    f = open(os.devnull, 'w')
    p = subprocess.Popen(saa_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    processes.append((p, f))
    # Wait for all processes to finish
    for p, f in processes:
        status = p.wait()
        f.close()

    for line in p.stdout:
        print(line[:-1].decode('utf-8'))