import numpy as np
import sys
sys.path.append("../")
from src.pairwise_fitter import Pairwise_fitter
import os
import argparse
import time

def listdir_nohidden(path):
    out = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            out.append(f)
    return out


# fit the ising model to each digit
def fit_digit(digit, seed, nsamples,\
              fname, inalldir_rel, 
              outdir_rel, exe_rel, method):
    print(fname.format(digit), seed, nsamples)
    mod = Pairwise_fitter(nsamples,inalldir_rel,fname.format(digit), outdir_rel)
    mod.setup(seed,input_spaced=False)
    mod.fit(method,exe_rel)


def main(nsamples, digit, method, train_full = "full"):
    seed = 1
    seed_plus = True
    if method == "ace":
        exe_rel="../ace_utils/ace"
    elif method == "rise":
        exe_rel="../ace_utils/rise"
        
    fname = train_full + "-images-unlabeled-{}"
    if train_full=="full":
        inalldir_rel="../data/INPUT_all/data/combined_data/"
    elif train_full=="train":
        inalldir_rel="../data/INPUT_all/data/traindata/"


    sample_size_dir = os.path.join(f"../data/OUTPUT_mod/data/{train_full}_sample_sizes/{method}/",str(nsamples))

    os.makedirs(sample_size_dir, exist_ok=True)
    if seed_plus:
        seed = 0 if seed is None else seed 

        # do not make new seed if lower seeds dont have our digit yet.
        # take the lowerst seed that is missing the digit and write output to there
        existing_seedslist = sorted([int(i) for i in listdir_nohidden(sample_size_dir)])
        print(listdir_nohidden(sample_size_dir))
        seed += len(existing_seedslist) 

        for ex_seed in existing_seedslist:
            test_dir = os.path.join(sample_size_dir,str(ex_seed))
            if not os.path.exists(os.path.join(test_dir, fname.format(digit))):
                seed = ex_seed
                break

        print("seed=",seed)
        
    outdir_rel=os.path.join(sample_size_dir,str(seed))
    os.makedirs(outdir_rel, exist_ok=True)


    start_time = time.time()
    fit_digit(digit=digit, seed=seed, nsamples=nsamples,\
               fname=fname, inalldir_rel=inalldir_rel, outdir_rel=outdir_rel, exe_rel=exe_rel, method=method)
    end_time = time.time()
    fitting_time = end_time - start_time
    with open(os.path.join(outdir_rel, fname.format(digit), 'fitting_time.txt',), 'w') as f:
        f.write(f"{fitting_time}")

    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit the ising model to the specified digit')
    parser.add_argument('--sample_s', type=int, help='Number of samples')
    parser.add_argument('--digit', type=int, help='Digit to fit')
    parser.add_argument('--method',type=str, help="ace or rise")
    parser.add_argument('--trainfull',type=str, help="full or train")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.sample_s, args.digit, args.method, args.trainfull)




