
import numpy as np

from src.classify import MCM_Classifier
import json
import os
from shutil import copytree
import src.plot as myplot




### --- FUNCTIONS FOR DATA SUBSETTING ---
def evaluate_subsample(sample_size,MCM_Classifier_init_args, all_data_path="../INPUT_all/data",
                        result_sample_sizes_dir="../OUTPUT/sample_sizes", comms_dir = "../OUTPUT/comms",estimator="add_smooth", seed=None,fname_start="train-", input_data_path="../INPUT/data"):
    """
    Generate sample_size number of samples and populate "../INPUT" folder. 
    Then fit the model to that data and save MCM and Counts from that model
      in a directory named after the sample size in the "../OUTPUT/sample_sizes" folder.

    :param sample_size: The number of images per class that should be used, if None then use all..
    :type sample_size: int
    :param all_data_path: The path to the data directory that will not be changed and where data is read from,
                            defaults to "../INPUT_all/data"
    :type all_data_path: str, optional
    :param result_sample_sizes_dir: The path to the output directory for saving the results,
                            defaults to "../OUTPUT/sample_sizes"
    :type result_sample_sizes_dir: str, optional
    :param comms_dir: directory of the communities after the current fitting
    """
    # subsample the data

    subsample_data(sample_size, all_data_path=all_data_path, seed=seed, fname_start=fname_start, input_data_path=input_data_path)
    # Fit new classifier object
    classifier = MCM_Classifier(*MCM_Classifier_init_args)
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000, estimator=estimator)


    # Save MCMS and Counts
    nwdir = os.path.join(result_sample_sizes_dir, str(sample_size))
    os.makedirs(nwdir, exist_ok=True)

    mcmdir = os.path.join(nwdir, "MCMs")
    countsdir = os.path.join(nwdir, "Counts")
    os.makedirs(mcmdir, exist_ok=True)
    os.makedirs(countsdir, exist_ok=True)


    # Append the number of files + 1 to the file names
    mcm_file_name = "MCMs_" + str(len(os.listdir(mcmdir))) + ".json"
    counts_file_name = "Counts_" + str(len(os.listdir(countsdir))) + ".json"

    # Save MCMS and Counts with the updated file names
    with open(os.path.join(mcmdir, mcm_file_name), 'w') as f:
        json.dump([arr.tolist() for arr in classifier.get_MCMs()], f, indent=2) 

    with open(os.path.join(countsdir, counts_file_name), 'w') as f:
        json.dump(classifier.get_Counts(), f, indent=2)


    # # Copy the new communities -> are also in MCM now
    # ncom = os.path.join(nwdir, "comms")
    # os.makedirs(ncom,exist_ok=True)
    # copytree(comms_dir, ncom,dirs_exist_ok=True)




    


def subsample_data(sample_size, all_data_path="../INPUT_all/data", input_data_path="../INPUT/data", seed=42,fname_start = "train-"):
    """Clear the input_data_path folder and fill it with samples from the all_data_path folder.
    
    :param sample_size: if None then take whole sample, otherwise provide integer of how many samples. Must be <= available samples.

    """
    rng = np.random.default_rng(seed)

    # Iterate over the files and delete the ones that start with "train-"
    for file in os.listdir(input_data_path):
        if file.startswith(fname_start):
            os.remove(os.path.join(input_data_path, file))

    # generate new input data 
    for file in os.listdir(all_data_path):
        if file.startswith(fname_start):
            print(os.path.join(input_data_path, file))
            inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
            np.savetxt(os.path.join(input_data_path, file), rng.choice(inp, sample_size,replace=False), fmt="%s")

#------------------------------ 


# def nudge_dataset(X, Y):
#     """
#     This produces a dataset 5 times bigger than the original one,
#     by moving the 8x8 images in X around by 1px to left, right, down, up
#     """
#     direction_vectors = [
#         [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
#     ]

#     def shift(x, w):
#         return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

#     X = np.concatenate(
#         [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
#     )
#     Y = np.concatenate([Y for _ in range(5)], axis=0)
#     return X, Y



# TOOLS FOR CONVERGENCE B PLOT -------------------------------


# XXX_sample => [sample size, run_index, cat_idx, icc_idx]
def load_counts_mcm(sample_sizes, letter, path_format = "../OUTPUT/sample_sizes_split_{}"):
    """For all sample sizes, load the counts and mcms for a specific digit and a pool of samples specifid by the letter."""
    samples_path = path_format.format(letter)

    fname = "Counts_"
    counts_sample = []
    for s, sample_size in enumerate(sample_sizes):
        counts_runs = []
        counts_path = os.path.join(samples_path, str(sample_size), "Counts")
        for i in range(len(os.listdir(counts_path))):
            with open(os.path.join(counts_path, fname+str(i)+ ".json")) as f:
                    counts_runs.append(json.load(f))
        
        counts_sample.append(counts_runs)

                    
    fname = "MCMs_"
    mcm_sample = []
    for s, sample_size in enumerate(sample_sizes):
        mcm_runs = []
        mcms_path = os.path.join(samples_path, str(sample_size), "MCMs")
        for i in range(len(os.listdir(mcms_path))):
            with open(os.path.join(mcms_path, fname+str(i)+ ".json")) as f:
                    # samples[sample_size] = json.load(f)
                    mcm_runs.append(json.load(f))
        mcm_sample.append(mcm_runs)


    return counts_sample, mcm_sample

# counts_sample, mcm_sample = load_counts_mcm(0,10,[10,100],"B")


# get sample seed=42 from dataset B
def recreate_dataset(sample_from_letter, sample_size:int, seed = 42):
    """Recreate the dataset A or B was build on."""
    all_data_path="../INPUT_all/data/combined_split_{}".format(sample_from_letter)
    file = "half-images-unlabeled-0.dat"
    sample_size_from_letter = sample_size # needs to be the same as build from sample size. We just show it the exact samples the other one is build on.
    rng = np.random.default_rng(seed=seed)
    inp = np.loadtxt(os.path.join(all_data_path,file), dtype="str")
    return rng.choice(inp, sample_size_from_letter,replace=False)

# sample_B = recreate_dataset("B", 10)


# average probability for each icc for observing the data B of the same size was build on
# do the same for observing A
def generate_counts_ranks_singlerun_singlesample(counts_sample, mcm_sample, see_data,run_idx, sample_idx, cat_idx = 0):
    """Generate the observed counts for iccs to observe the "see_data" sample. Also returns the ranks of the iccs.
    That is it returns a list of count distributions over samples for every icc.
    """
    sum_of_count = np.sum(counts_sample[sample_idx][run_idx][cat_idx][0])
    data = np.array([[int(s) for s in state] for state in see_data])
    nr_icc = len(counts_sample[sample_idx][run_idx][cat_idx])
    counts_observe_B = np.empty((nr_icc, see_data.shape[0]))
    for icc in range(nr_icc):
        counts_observe_B[icc,:] = (myplot.calc_p_icc_single(data,counts_sample[sample_idx][run_idx][cat_idx],121,mcm_sample[sample_idx][run_idx][cat_idx],icc))

    ranks = np.genfromtxt(mcm_sample[sample_idx][run_idx][cat_idx],dtype=int,delimiter=1).sum(axis=1)

    return counts_observe_B, ranks


def counts_to_prob(icc_pdf, rank,sum_of_count, add_smooth=True):
    alpha = 1
    if add_smooth:
        zero_f =  lambda x: (x+(1/(2**rank)))/ (sum_of_count+1) 
    else:
        zero_f = lambda x: x/sum_of_count

    return np.apply_along_axis(zero_f,0,icc_pdf)

# calculate probabilities P(MCM_A see data_B)
def probs_mean_std(counts_observe_X, ranks, sum_of_count, data_size,n_icc,add_smooth):
    """Calculate mean and std probability of mcm to observe some data over samples.
    That is mean over runs (P(mcm_digitx_samplesizex observes data X). 
    e.g., mcm digit 0 using 100 samples from pool A to build it sees 100 samples from pool B"""

    probs = np.empty((n_icc,data_size))
    for i in range(len(ranks)):
        probs[i,:] = counts_to_prob(counts_observe_X[i],ranks[i], sum_of_count, add_smooth)
    probs_mcm = np.product(probs,axis=0)

    return np.mean(probs_mcm), np.std(probs_mcm)



# main function for convergence B
def letter_means_stds(letter, sample_sizes, nr_runs, digit,recreate_letter,add_smooth, data_size="same",):
    counts_sample, mcm_sample = load_counts_mcm(sample_sizes,letter)


    ms_all = np.empty((len(sample_sizes),nr_runs,2))
    for sample_idx, sample_size in enumerate(sample_sizes):
        m_s_run = np.empty((nr_runs,2))
        if data_size =="same":
            data_size = sample_size
    
        for run_idx in range(nr_runs):
            n_icc = len(counts_sample[sample_idx][run_idx][digit])
            sample_recreate = recreate_dataset(recreate_letter, int(data_size))

            counts_observe_X, ranks = generate_counts_ranks_singlerun_singlesample(counts_sample, mcm_sample, sample_recreate, run_idx, sample_idx)
            sum_of_count = np.sum(counts_sample[sample_idx][run_idx][0][0])
            # mean probabilities over 100 B samples for some mcm
            m,s = probs_mean_std(counts_observe_X, ranks, sum_of_count,data_size, n_icc, add_smooth=add_smooth) 
            ms_all[sample_idx, run_idx, :] = [m,s]

    return ms_all

def load_test_data(digit = 0):
        all_data_path="../INPUT_all/data/testdata_separated"
        file = "test-images-unlabeled-{}.dat".format(digit)
        return np.loadtxt(os.path.join(all_data_path,file), dtype="str")



#### PIXELWISE EVIDENCE
import math

def evidence_iccs(Counts, MCMs, mcm_idx):
    """Calculate the evidence for each icc in an MCM using the count distribution of the g* parameters/ ML estimate.
      Return an array of evidences. The sum of that array is the MCM evidence.

    :param Counts: Return value of classifier.get_Counts(). Unnormalized probability distribution for all MCM for all ICC.  
    :type Counts: np.ndarray of shape[category,icc,possible_states]
    :param MCMs: Return value of classifier.get_MCMs(). MCMs for all categories.
    :type MCMs: nested list of shape[category,icc] of binary strings.
    :param mcm_idx: which mcm to calculate the evidence for.
    :type mcm_idx: int < #categories
    :return: np.ndarray of evidences for each icc. ICC are identified by index
    :rtype: np.ndarray of shape [icc_evidences]
    """
    N = np.array(Counts[0][0]).sum().astype(int) # sample size == sum of observed states
    count_mcm = Counts[mcm_idx]
    evidence = np.zeros(len(count_mcm)) # nr iccs
    log_sqrt_pi = math.log(math.sqrt(math.pi))
    for idx, icc in enumerate(count_mcm):
        rank = MCMs[mcm_idx][idx].count("1")
        evidence[idx] += math.lgamma(2**(rank-1)) - math.lgamma(N + 2**(rank-1)) # middle part of equation 8 in Mulatier_2020
        for pattern in Counts[mcm_idx][idx]: # last part of equation 8
            evidence[idx] += math.lgamma(pattern+.5) - log_sqrt_pi
    return evidence

# evidence_iccs(Counts,MCMs,2)

def pixelwise_evidence(evidence_iccs,N,single_mcm):
    icc_pixels = [icc.count("1") for icc in single_mcm]
    return evidence_iccs / np.log(2) / N / icc_pixels



##### Evidence Part 2
def evidence_on_data(single_mcm, data):
    """Calculate the evidence of a partitioning ("single_mcm") on some (possibly new) data.

    :param single_mcm: Each binary string is an icc state.
    :type single_mcm: np.array 1D with dtype string
    :param data: Dataset to calcualte evidence on. result from np.loadtext(dtype=str)
    :type data: np.array 1D with dtype string
    """
    mcm_gen = np.array([[int(s) for s in state] for state in single_mcm])
    data_gen = np.array([[int(s) for s in state] for state in data])
    N = len(data)
    nr_iccs = len(single_mcm)

    evidence = np.zeros(nr_iccs)
    log_sqrt_pi = math.log(math.sqrt(math.pi))
    for icc_idx, icc in enumerate(single_mcm):
        rank = icc.count("1")
        C_icc = data_gen[:,mcm_gen[icc_idx,:] == 1]
        counts = np.unique(C_icc, axis=0, return_counts=True)[1]


        evidence[icc_idx] += math.lgamma(2**(rank-1)) - math.lgamma(N + 2**(rank-1)) # middle part of equation 8 in Mulatier_2020
        for k in counts: # last part of equation 8
            evidence[icc_idx] += math.lgamma(k+.5) - log_sqrt_pi

        mcm_evidence = np.sum(evidence) # - N*(0)*np.log(2)
    return mcm_evidence