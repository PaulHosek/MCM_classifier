import numpy as np
import copy
from src.classify import MCM_Classifier
import json
import os
from shutil import copytree
import src.plot as myplot
import src.loaders as loaders
# 


### --- FUNCTIONS FOR DATA SUBSETTING ---
def evaluate_subsample(sample_size,MCM_Classifier_init_args, all_data_path="../data/INPUT_all/data",
                        result_sample_sizes_dir="../data/OUTPUT/mcm/sample_sizes", comms_dir = "../data/OUTPUT/mcm/comms",estimator="add_smooth", seed=None,fname_start="train-", input_data_path="../data/INPUT/data", seed_plus=False):
    """
    Generate sample_size number of samples and populate "../data/INPUT" folder. 
    Then fit the model to that data and save MCM and Counts from that model
      in a directory named after the sample size in the "../OUTPUT/sample_sizes" folder.

    :param sample_size: The number of images per class that should be used, if None then use all..
    :type sample_size: int
    :param all_data_path: The path to the data directory that will not be changed and where data is read from,
                            defaults to "..data/INPUT_all/data"
    :type all_data_path: str, optional
    :param result_sample_sizes_dir: The path to the output directory for saving the results,
                            defaults to "..data/OUTPUT/mcm/sample_sizes"
    :type result_sample_sizes_dir: str, optional
    :param comms_dir: directory of the communities after the current fitting
    :param seed_plus: if true, then do seed += nr runs for that sample done: e.g.,
    """

    # output dirs
    nwdir = os.path.join(result_sample_sizes_dir, str(sample_size))
    os.makedirs(nwdir, exist_ok=True)
    mcmdir = os.path.join(nwdir, "MCMs")
    countsdir = os.path.join(nwdir, "Counts")
    os.makedirs(mcmdir, exist_ok=True)
    os.makedirs(countsdir, exist_ok=True)

    if seed_plus:
        seed = 0 if seed is None else seed
        seed += len(os.listdir(mcmdir))
        print("seed=", seed)

    # subsample the data
    subsample_data(sample_size, all_data_path=all_data_path, seed=seed, fname_start=fname_start, input_data_path=input_data_path)
    # Fit new classifier object
    classifier = MCM_Classifier(*MCM_Classifier_init_args)
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000, estimator=estimator)





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




    


def subsample_data(sample_size, all_data_path="../data/INPUT_all/data", input_data_path="../data/INPUT/data", seed=42,fname_start = "train-"):
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
            # print(os.path.join(input_data_path, file))
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
def load_counts_mcm(sample_sizes, letter, path_format = "../data/OUTPUT/mcm/sample_sizes_split_{}", ):
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
def recreate_dataset(sample_from_letter,digit, sample_size:int, seed = 42,fname_format= "half-images-unlabeled-{}.dat", fname_start="half-", all_data_path="../data/INPUT_all/data/combined_split_{}", input_data_path = "../data/INPUT/data/"):
    """Recreate the dataset A or B was build on."""
    all_data_path = all_data_path.format(sample_from_letter)
     
    sample_size_from_letter = sample_size # needs to be the same as build from sample size. We just show it the exact samples the other one is build on.
    subsample_data(sample_size_from_letter, all_data_path=all_data_path, seed=seed, fname_start=fname_start, input_data_path=input_data_path)


    res = loaders.load_data(os.path.join(input_data_path,fname_format.format(digit)))
    res = ["".join(i) for i in res.astype(str)]
    return np.array(res)

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
        print("nested")
        for run_idx in range(nr_runs):
            n_icc = len(counts_sample[sample_idx][run_idx][digit])
            seed_run = run_idx+1 # for A,B

            sample_recreate = recreate_dataset(recreate_letter,digit, int(data_size), seed=seed_run)
            counts_observe_X, ranks = generate_counts_ranks_singlerun_singlesample(counts_sample, mcm_sample, sample_recreate, run_idx, sample_idx)

            
            sum_of_count = np.sum(counts_sample[sample_idx][run_idx][0][0])
            # mean probabilities over 100 B samples for some mcm
            m,s = probs_mean_std(counts_observe_X, ranks, sum_of_count,data_size, n_icc, add_smooth=add_smooth) 
            ms_all[sample_idx, run_idx, :] = [m,s]


            # test if evidence the same: fitted counts = observed counts
            # my_counts = counts_sample[sample_idx][:nr_runs] # test
            # mcms = mcm_sample[sample_idx][:nr_runs]
            # print(len(my_counts[run_idx]),len(mcms))
            # per_icc = np.sum(evidence_iccs(my_counts[run_idx], mcms[run_idx],digit))#/ my_sample_size
            # ev = evidence_on_data(mcms[run_idx][digit], sample_recreate)
            # if not ev == per_icc and recreate_letter == letter:
            #     raise KeyboardInterrupt
            # print(ev == per_icc, ev, per_icc) 

    return ms_all

def load_test_data(digit = 0, all_data_path="../data/INPUT_all/data/testdata_separated"):
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
        for pattern in icc: # last part of equation 8
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
        # counts = counts[np.argsort([int("".join(i),base=2) for i in configs.astype(str)])] # order counts by integer representation of config


        evidence[icc_idx] += math.lgamma(2**(rank-1)) - math.lgamma(N + 2**(rank-1)) # middle part of equation 8 in Mulatier_2020
        for k in counts: # last part of equation 8
            evidence[icc_idx] += math.lgamma(k+.5) - log_sqrt_pi

        mcm_evidence = np.sum(evidence) # - N*(0)*np.log(2)
    return mcm_evidence


def probabilities_gstar(single_mcm,counts_gstar, data,fitting_sample_size, smooth=True, return_distr_icc=False):
    """Calculate the probability distribution at g* of a partitioning ("single_mcm") on some (possibly new) data.

    :param single_mcm: Each binary string is an icc state.
    :type single_mcm: np.array 1D with dtype string
    :param data: Dataset to calcualte evidence on. result from np.loadtext(dtype=str)
    :type data: np.array 1D with dtype string
    :param fitting_sample_size: nr samples used to build the mcm
    :param return_distr_icc: instead of multiplying icc, return array of max size icc filled with first k entries with the icc, all other with -1.
    :type return_distr_icc: 1d np.array of size nr spins (i.e., 121). filled with p_icc if icc exites, else -1.
    :returns: 1D np.array of len(data) with the final probability for observing each image
    """
    mcm_gen = np.array([[int(s) for s in state] for state in single_mcm])
    data_gen = np.array([[int(s) for s in state] for state in data])

    
    N = len(data)

    distr = np.ones(len(data))
    if return_distr_icc:
        distr = np.full((121, len(data)), fill_value = -1.0)    
    # calcualte probability of MCM
    for icc_idx, icc in enumerate(single_mcm):
        # calculate probabilities of icc seeing each of the states in data
        rank = icc.count("1")
        counts_icc = np.array(counts_gstar[icc_idx]) # e.g., [188,12,0,0] if rank = 2 and 200 samples
        
        observables = data_gen[:,mcm_gen[icc_idx,:] == 1]
        obs_states = np.apply_along_axis(lambda i: int("".join(i),base=2), 1, observables.astype(str))


        kba = counts_icc[obs_states]
        if return_distr_icc:
            if smooth:
                distr[icc_idx,:] = (kba+(1/(2**rank)))/(fitting_sample_size+1)
            else:
                distr[icc_idx, :] = kba/fitting_sample_size
        else:
            if smooth:
                distr *= (kba+1/(2**rank))/(fitting_sample_size+1)
            else:
                distr *= kba/fitting_sample_size
            

    return distr

def normalised_signed_distance_decisionbound(probs, own_cat, other_cat):
    """Generate distances to decision boundary array to quantiy indicative icc.



    :param probs: _description_
    :type probs: 2d np array of shape [N, ≥2]
    :param own_cat: digit that the mcm was fitted on (e.g., 3)
    :type own_cat: int 
    :param other_cat: digit to test against (e.g., 5)
    """
    assert own_cat <= probs.shape[1] and other_cat <= probs.shape[1], "IndexError: own_cat or other_cat are out of bounds of the probs array."

    x_coords, y_coords = zip(*[(x, y) for x, y in zip(probs[:, own_cat], probs[:, other_cat],)])
    distances = np.array(x_coords) - np.array(y_coords) / np.sqrt(2) # signed distance
    x0, y0 = 0, 1
    max_dist = np.abs(x0 - y0) / np.sqrt(2)
    return distances/max_dist


def norm_distribution_distance(dist_a,dist_b):
    """
    Distance between two probability distributions.
    Implements logic based on to |mu_1 - mu_2| > (sigma_1 + sigma_2).
    Formula used:
     [|mu_1 - mu_2| / (sigma_1 + sigma_2)] 

     Positive if mean difference is larger than twice the standard deviation.
     Negative if smaller.
     0 if equal.
    

    :param dist_a: The first probability distribution.
    :type dist_a: numpy.ndarray
    :param dist_b: The second probability distribution.
    :type dist_b: numpy.ndarray
    :return: The distance between the two probability distributions.
    :rtype: float
    """
    
    return np.divide(np.abs(np.mean(dist_a) - np.mean(dist_b)), (np.std(dist_a) +  np.std(dist_b)))



def total_variation_distance(dist_a, dist_b):
    """
    Calculates the total variation distance (TV)/ normalized L1 distance between two probability distributions.
    Averaged per size of distr.
    
    :param dist_a: The first probability distribution.
    :type dist_a: numpy.ndarray
    :param dist_b: The second probability distribution.
    :type dist_b: numpy.ndarray
    :return: The total variation distance between the two distributions.
    :rtype: float
    """
    return 0.5 * np.sum(np.abs(dist_a - dist_b)) / len(dist_a)



def get_complete_testprobs(mcms_samplesizes,counts_samplesizes,sample_sizes,n_runs,nr_digits=10,nr_mcms=10, maxnr_icc=121, smooth=True):
    """
    Computes the g* probabilities of every individual icc for each sample, for each run, for each digit, for each mcm.
    returns test_mcm np array. Shape: (nr_mcms, nr_sample_sizes, n_runs,nr_digits,maxnr_icc, nr_testimg) with , fill_value=-1.0
    returns test_probs np array. Shape: (nr_mcms, len(sample_sizes), n_runs,nr_digits,maxnr_icc) with  fill_value="", dtype="<U121"

    :param mcms_samplesizes: List of sample sizes used to build the MCMs.
    :type mcms_samplesizes: list
    :param counts_samplesizes: List of sample sizes used to calculate the counts.
    :type counts_samplesizes: list
    :param sample_sizes: List of sample sizes for which to compute the probabilities.
    :type sample_sizes: list
    :param n_runs: Number of runs.
    :type n_runs: int
    :param nr_digits: Number of digits.
    :type nr_digits: int, optional
    :param nr_mcms: Number of MCMs.
    :type nr_mcms: int, optional
    :param maxnr_icc: Maximum number of iccs.
    :type maxnr_icc: int, optional
    :param smooth: Whether to apply smoothing to the probabilities, defaults to True.
    :type smooth: bool, optional

    :return: The test_mcm np array. Shape: (nr_mcms, nr_sample_sizes, n_runs,nr_digits,maxnr_icc, nr_testimg) with , fill_value=-1.0
    and the test_probs array. Shape: (nr_mcms, len(sample_sizes), n_runs,nr_digits,maxnr_icc) with  fill_value="", dtype="<U121"

    :rtype: tuple
    """

    test_probs = np.full((nr_mcms, len(sample_sizes), n_runs,nr_digits,maxnr_icc, len(load_test_data(digit=0))), fill_value=-1.0)
    test_mcms = np.full((nr_mcms, len(sample_sizes), n_runs,nr_digits,maxnr_icc), fill_value="", dtype="<U121")
    for mcm_digit in range(nr_digits):

        for test_digit in range(nr_digits):
            test_data = load_test_data(digit=test_digit)

            for sample_size_idx, sample_size in enumerate(sample_sizes):

                mcms = mcms_samplesizes[sample_size_idx][:n_runs]
                counts_gstar = counts_samplesizes[sample_size_idx][:n_runs]
                    
                for run_idx, mcm in enumerate(mcms):
                    test_probs[mcm_digit][sample_size_idx][run_idx][test_digit] = probabilities_gstar(mcm[mcm_digit], counts_gstar[run_idx][mcm_digit], test_data, sample_size,smooth=smooth,return_distr_icc=True)
                    test_mcms[mcm_digit][sample_size_idx][run_idx][test_digit][:len(mcm[mcm_digit])] = np.array(mcm[mcm_digit],dtype=str)
    return test_mcms, test_probs

def distmap_from_testprobs(test_probs,test_mcms,digit_pair, mcm_idx, sample_idx, run_idx, return_comms = False, return_iccdata = False, return_avg_icc_prob=False,return_dists=False):

    icc_data = test_probs[mcm_idx,sample_idx, run_idx]

    # get into format for decision boundary distance function
    nr_icc = np.argmin(icc_data != -1, axis =1)[0,0] # remove icc rows of surplus icc, all the same
    icc_data = icc_data[:,:nr_icc,:]
    icc_data = np.transpose(icc_data, (1, 2, 0)) 
    avg_icc_prob = icc_data.mean(axis=1) # over test samples
    dists = normalised_signed_distance_decisionbound(avg_icc_prob,digit_pair[0],digit_pair[1])
    # find mcm and build map
    intr_mcm = test_mcms[mcm_idx,sample_idx, run_idx][0,:nr_icc] # get rid of unused rows, mcm_identical for every icc
    comms = myplot.generate_icc_comms_map(intr_mcm)
    dist_map = dists[comms] 
    # plt.scatter(avg_prob[:,cat_a], avg_prob[:,cat_b])
    # plt.imshow(dist_map)

    out = [dist_map]
    if return_comms:
         out.append(comms)
    if return_iccdata:
        out.append(icc_data)
    if return_avg_icc_prob:
        out.append(avg_icc_prob)
    if return_dists:
        out.append(dists)

    return tuple(out)


def get_all_byk_pair(test_probs, test_mcms, digit_pair,sample_idx,run_idx):
    """Compare two MCM fitted on the digit_pair digits
      on how many of the top ICC they need to differentiate the digit pair.

    
    Generates all_byk_pair list of np arrays.
    Each element of the list is an mcm. Within each list, there is a np array of shape (nicc,ntestimg,digitpair).
    The first index (nicc) is the cumprod of the top k icc for that binary difference.
    The index of the last dimension is in the same order as the provided digit pair.
    
    Generates all_byk_modspin list of 1d np arrays.
    Each eleent of the list is an mcm. Each np array is the commulative sum of SPINS in the k iccs modelled.
    This list can be used to adjust the probabilities between k.

    :param test_probs: result array from paper_utils.get_complete_testprobs. Probability on test set per icc, MCM, digit, run, sample size.
    :type test_probs: np.ndarray
    :param test_mcm: result array from paper_utils.get_complete_testprobs. MCMs for every run, digit, sample size.
    :param digit_pair: the digits to test
    :type digit_pair: tuple or list of len 2
    :param sample_idx: which sample size to use. Provide index, not sample size.
    :type sample_idx: int
    :param run_idx: Which run to use. Provide index.
    :type run_idx: int
    """
    all_byk_pair = []
    all_byk_modspin = []
    for mcm_idx in digit_pair:
        _, comms,icc_data,dists = distmap_from_testprobs(test_probs, test_mcms, digit_pair, mcm_idx, sample_idx,run_idx, return_iccdata=True,return_dists=True, return_comms=True)
        ord_distidcs = np.argsort(dists)[::-1]
        by_k = np.cumprod(icc_data[ord_distidcs],axis=0)[:,:,digit_pair]
        all_byk_pair.append(by_k)

            # compute the number of spins in each model
        icc_sizes = np.unique(comms,return_counts=True)[1]
        modelled_spins = np.cumsum(icc_sizes[ord_distidcs])
        all_byk_modspin.append(modelled_spins)
    return all_byk_pair, all_byk_modspin



def adjust_smaller_icc(all_byk,all_byk_modspin):
    """For a comparision of two MCM over K icc, at each k adjust the size of
      the smaller sub-MCM to the size of the larger one by adding unmodelled spins.

    :param all_byk: Probabilities for ICC in two models over data.
    :type all_byk: list of two np arrays of shape (nicc, nimages, seedigit(of 2))
    :param all_byk_modspin: number spins in each sub-mcm
    :type all_byk_modspin: 1d np.array
    """
    c_all_byk = copy.deepcopy(all_byk)  # create a copy of all_byk

    nicc_mods = [len(i) for i in all_byk_modspin] # 33, 30
    mods_long_short = np.argsort(nicc_mods)[::-1]
    nspin_diff = all_byk_modspin[mods_long_short[0]][:np.min(nicc_mods)] - all_byk_modspin[mods_long_short[1]][:np.min(nicc_mods)]

    nspin_diff = np.append(nspin_diff,(  all_byk_modspin[mods_long_short[0]][np.min(nicc_mods):] - all_byk_modspin[mods_long_short[1]][-1]))

    # print(( 121 - all_byk_modspin[long_mod][np.min(nicc_mods):]))
    assert len(nspin_diff) == np.max(nicc_mods)
    for i,r_diff in enumerate(nspin_diff):
        # if model 0 more spins than model 1 for that k, adjust model 1 probability
        if r_diff > 0:
            c_all_byk[mods_long_short[1]][i,...] *= 1/(2**np.abs(r_diff))
        elif r_diff < 0:
            c_all_byk[mods_long_short[0]][i,...] *= 1/(2**np.abs(r_diff))

    return c_all_byk



def partition_to_str(mcm):
    """Take a partition map labeling 11x11 array and return the original string representation"""
    nr_icc = mcm.max()
    out = []

    for icc in range(1, nr_icc+1):
        idx = np.argwhere(mcm.flatten()==icc).flatten()
        x = np.zeros(121,dtype=int)
        x[idx] = 1
        out.append("".join(map(str, x)))
    return np.array(out, dtype=str)