#include "header.h"

void greedy_merging(Partition &p_struct) {

    cout << "- running greedy merging algorithm on " << p_struct.nc << " communities\n" << endl; 

    double best_delta = 1;
    map<__uint128_t, double> calculated_evidence;
    unsigned int depth = 0;
    double delta_evidence, unmerged_evidence, merged_evidence;
    double best_merge;

    unsigned int best_i, best_j, last_i;
    best_i = best_j = last_i = 0;

    while (best_delta > 0) {

        best_delta = 0;
        best_merge = 0;
        
        __uint128_t best_community = 0;

        for (unsigned int i = 0; i < p_struct.n; i++){
            for (unsigned int j = i + 1; j < p_struct.n; j++){

                __uint128_t ci = p_struct.current_partition[i];
                __uint128_t cj = p_struct.current_partition[j];
                __uint128_t cij =  ci + cj;

                if ((bit_count(ci) == 0) || (bit_count(cj) == 0)) {continue;}

                if ((depth == 0) || (i == last_i) || (j == last_i)) {
                    unmerged_evidence = icc_evidence(ci, p_struct) + icc_evidence(cj, p_struct);
                    merged_evidence = icc_evidence(cij, p_struct);
                    delta_evidence = merged_evidence - unmerged_evidence;
                    __uint128_t identifier = (ONE << i) + (ONE << j);
                    calculated_evidence[identifier] = delta_evidence;           
                } else {
                    __uint128_t identifier = (ONE << i) + (ONE << j);
                    delta_evidence = calculated_evidence[identifier];
                }

                if (delta_evidence > best_delta) {
                    best_delta = delta_evidence;
                    best_merge = icc_evidence(cij, p_struct);
                    best_i = i;
                    best_j = j;
                    best_community = cij;
                }

            }

        }

        // perform merge
        if (best_delta > 0) {

            cout << "merging communities: " << best_i << " and " << best_j << " | delta log-e: " << best_delta << endl;
            cout << "c1:  " << int_to_bitstring(p_struct.current_partition[best_i], p_struct.n) << endl;
            cout << "c2:  " << int_to_bitstring(p_struct.current_partition[best_j], p_struct.n) << endl;
            cout << "c12: " << int_to_bitstring(best_community, p_struct.n) << endl;

            last_i = best_i;
            depth++;

            p_struct.current_partition[best_i] = best_community;
            p_struct.current_partition[best_j] = 0;
            p_struct.best_partition[best_i] = best_community;
            p_struct.best_partition[best_j] = 0;    
            p_struct.partition_evidence[best_i] = best_merge;
            p_struct.partition_evidence[best_j] = 0;
            p_struct.occupied_partitions -= (ONE << best_j);
            p_struct.current_log_evidence += best_delta;
            p_struct.best_log_evidence = p_struct.current_log_evidence;

            cout << "best log-evidence: " << p_struct.best_log_evidence << "\n" << endl;
            p_struct.nc -= 1;

            // update communities of size >= 2 (for split & switch)
            if ((p_struct.occupied_partitions_gt2_nodes & (ONE << best_j)) == (ONE << best_j)){
                p_struct.occupied_partitions_gt2_nodes -= (ONE << best_j);
            }
            if ((p_struct.occupied_partitions_gt2_nodes & (ONE << best_i)) == 0){
                p_struct.occupied_partitions_gt2_nodes += (ONE << best_i);
            }   
        }       
    }
}