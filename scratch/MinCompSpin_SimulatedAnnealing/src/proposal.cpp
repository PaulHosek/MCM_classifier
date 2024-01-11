#include "header.h"

void merge_partition(Partition &p_struct){

	if (p_struct.nc <= 1){return;} // can't merge one community

	// choose two valid random communities
	unsigned int p1 = randomBitIndex(p_struct.occupied_partitions);
	unsigned int p2 = randomBitIndex(p_struct.occupied_partitions - (ONE << p1));

	// calculate log-evidence of merged community
	__uint128_t merged_community = p_struct.current_partition[p1] + p_struct.current_partition[p2];
	double merged_evidence = icc_evidence(merged_community, p_struct);
	double delta_log_evidence = merged_evidence - p_struct.partition_evidence[p1] - p_struct.partition_evidence[p2];
	
	// metropolis acceptance probability
	double p = exp(delta_log_evidence/p_struct.T);
	double u = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

	if (p > u){
		p_struct.current_partition[p1] = merged_community;
		p_struct.current_partition[p2] = 0;
		p_struct.partition_evidence[p1] = merged_evidence;
		p_struct.partition_evidence[p2] = 0;
		p_struct.occupied_partitions -= (ONE << p2); // remove empty community 
		p_struct.current_log_evidence += delta_log_evidence; // update total log-evidence
		p_struct.nc -= 1;

		// update communities of size >= 2 (for split & switch)
		if ((p_struct.occupied_partitions_gt2_nodes & (ONE << p2)) == (ONE << p2)){
			p_struct.occupied_partitions_gt2_nodes -= (ONE << p2);
		}
		if ((p_struct.occupied_partitions_gt2_nodes & (ONE << p1)) == 0){
			p_struct.occupied_partitions_gt2_nodes += (ONE << p1);
		}	

	}

}

void split_partition(Partition &p_struct){

	if (p_struct.nc == p_struct.n){return;} // can't split independent communities
	if (p_struct.occupied_partitions_gt2_nodes == 0){return;} // can't split communities of size 1

	// choose random valid community
	unsigned int p1 = randomBitIndex(p_struct.occupied_partitions_gt2_nodes);
	__uint128_t community = p_struct.current_partition[p1];

	__uint128_t mask = random_128_int(p_struct.n);
	__uint128_t c1 = (community & mask);
	__uint128_t c2 = (community & (~mask));

	// masking shouldn't assign all nodes to one community
	while ((bit_count(c1) == 0) || (bit_count(c2) == 0)){
		mask = random_128_int(p_struct.n);
		c1 = (community & mask);
		c2 = (community & (~mask));
	}

	double log_evidence_1 = icc_evidence(c1, p_struct);
	double log_evidence_2 = icc_evidence(c2, p_struct);
	double delta_log_evidence = log_evidence_1 + log_evidence_2 - p_struct.partition_evidence[p1];
	double p = exp(delta_log_evidence/p_struct.T);
	double u = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);


	if (p > u){
		// find empty spot for second split community
		unsigned int p2 = randomBitIndex(~p_struct.occupied_partitions - p_struct.unused_bits);

		p_struct.current_partition[p1] = c1;
		p_struct.partition_evidence[p1] = log_evidence_1;
		p_struct.current_partition[p2] = c2;
		p_struct.partition_evidence[p2] = log_evidence_2;
		p_struct.occupied_partitions += (ONE << p2);
		p_struct.current_log_evidence += delta_log_evidence;
		p_struct.nc += 1;

		if (bit_count(c1) <= 1){
			p_struct.occupied_partitions_gt2_nodes -= (ONE << p1);
		}
		if (bit_count(c2) >= 2){
			p_struct.occupied_partitions_gt2_nodes += (ONE << p2);
		}		

	}

}

void switch_partition(Partition &p_struct){

	if (p_struct.nc <= 1){return;} // can't switch node to same community 
	if (p_struct.nc == p_struct.n){return;} // switching independent nodes doesn't change anything
	if (p_struct.occupied_partitions_gt2_nodes == 0){return;} // don't create empty partitions

	unsigned int p1 = randomBitIndex(p_struct.occupied_partitions_gt2_nodes);
	unsigned int p2 = randomBitIndex(p_struct.occupied_partitions - (ONE << p1));
	__uint128_t c1 = p_struct.current_partition[p1];
	unsigned int node = randomBitIndex(c1);
	
	__uint128_t c2 = p_struct.current_partition[p2];
	__uint128_t new_c1 = c1 - (ONE << node); // remove from community 1
	__uint128_t new_c2 = c2 + (ONE << node); // add to community 2

	double log_evidence_1 = icc_evidence(new_c1, p_struct);
	double log_evidence_2 = icc_evidence(new_c2, p_struct);
	double delta_log_evidence = log_evidence_1 + log_evidence_2 - p_struct.partition_evidence[p1] - p_struct.partition_evidence[p2];
	double p = exp(delta_log_evidence/p_struct.T);
	double u = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

	if (p > u){
		p_struct.current_partition[p1] = new_c1;
		p_struct.current_partition[p2] = new_c2;
		p_struct.partition_evidence[p1] = log_evidence_1;
		p_struct.partition_evidence[p2] = log_evidence_2;
		p_struct.current_log_evidence += delta_log_evidence;

		if ((bit_count(c1) >= 2) && (bit_count(new_c1) <= 1)){
			p_struct.occupied_partitions_gt2_nodes -= (ONE << p1);
		} 
		if ((bit_count(c2) <= 1) && (bit_count(new_c2) >= 2)){
			p_struct.occupied_partitions_gt2_nodes += (ONE << p2);
		} 

	}

}