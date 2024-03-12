#include "header.h"

void parse_community(Partition &p_struct, __uint128_t community, int i){

	p_struct.current_partition[i] = community;
	p_struct.partition_evidence[i] = icc_evidence(community, p_struct);
	p_struct.current_log_evidence += p_struct.partition_evidence[i];
	p_struct.occupied_partitions += (ONE << i);
	p_struct.nc++;
	if (bit_count(community) >= 2){
		p_struct.occupied_partitions_gt2_nodes += (ONE << i);
	}

}


void random_partition(Partition &p_struct) {

	cout << "- starting from random partition" << endl;

	__uint128_t community;
	__uint128_t assigned = 0;

	int i = 0;

	while(bit_count(assigned) < p_struct.n) {

		community = random_128_int(p_struct.n);
		community = community - (assigned & community);
		assigned += community;

		if (bit_count(community) > 0) {

			parse_community(p_struct, community, i);

			cout << "- generated community: " << int_to_bitstring(community, p_struct.n) << endl;
			cout << "- log-evidence: " << p_struct.partition_evidence[i] << endl;
			//cout << "Assigned nodes: " << int_to_bitstring(assigned, p_struct.n) << endl;
			cout << endl;
			i++;
		}
	}

	p_struct.best_log_evidence = p_struct.current_log_evidence;
	p_struct.best_partition = p_struct.current_partition;

	cout << "- generated " << p_struct.nc << " communities" << endl;
	cout << "- initial log-evidence: " << p_struct.current_log_evidence << "\n" << endl;

}

void independent_partition(Partition &p_struct) {

	__uint128_t community;

	for (unsigned int i = 0; i < p_struct.n; i++) {

		community = (ONE << i);
		parse_community(p_struct, community, i);

		// cout << "New community: " << int_to_bitstring(community, p_struct.n) << endl;

	}

	p_struct.best_log_evidence = p_struct.current_log_evidence;
	p_struct.best_partition = p_struct.current_partition;

	cout << "- starting from independent partition" << endl;
	cout << "- initial log-evidence: " << p_struct.current_log_evidence << "\n" << endl;
}

void load_partition(Partition &p_struct, string pname) {

	p_struct.current_log_evidence = 0;

	string fpath = "../input/comms/" + pname + ".dat";
	string line;
	ifstream comm_file(fpath);
	__uint128_t community;
	int i = 0;
	while(getline(comm_file, line)){
		community = string_to_int(line, p_struct.n);
		parse_community(p_struct, community, i);
		i++;
	}

	comm_file.close();

	p_struct.best_log_evidence = p_struct.current_log_evidence;
	p_struct.best_partition = p_struct.current_partition;
	
	cout << "- loaded " << p_struct.nc << " communities" << endl;
	cout << "- initial log-evidence: " << p_struct.current_log_evidence << "\n" << endl;

}