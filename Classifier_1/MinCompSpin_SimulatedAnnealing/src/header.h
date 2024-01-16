#include <iostream>
#include <vector>
#include <bitset>
#include <random>
#include <map>
#include <unordered_map>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

const __uint128_t ONE = 1; // 128-bit representation of 1
const __uint128_t ZERO = 0;
const __uint128_t NOT_ZERO = ~0; // 128-bit integer containing only ones
const double EPSILON = 1e-4; // minimum change in log-evidence 
const double SQRT_PI = lgamma(0.5);

struct Partition {

	Partition(const unsigned int &n_) : n(n_) {}

	unsigned int n; // number of variables
	__uint128_t unused_bits = NOT_ZERO - ((ONE << n) - 1); // ones in the unused bits - used for fast location of empty partitions

	double T; // annealing temperature
	double current_log_evidence = 0;
	double best_log_evidence = 0;

	unsigned int N = 0; // number of samples in dataset
	unsigned int nc = 0; // current number of communities 

	__uint128_t occupied_partitions = 0; // occupied communities (with at least one node)
	__uint128_t occupied_partitions_gt2_nodes = 0; // communities with at least two nodes (for split and switch)
	
	vector<pair<__uint128_t, unsigned int>> data;
	vector<__uint128_t> current_partition = vector<__uint128_t>(n);
	vector<__uint128_t> best_partition = vector<__uint128_t>(n);
	vector<double> partition_evidence = vector<double>(n); // log-evidence for each community 

	map<__uint128_t, double> evidence_memo;
};

// function declarations
bool DoubleSame(double a, double b);

// bit manipulation stuff
unsigned int bit_count(__uint128_t number);
unsigned int randomBitIndex(__uint128_t number);
string int_to_bitstring(__uint128_t number, unsigned int r);
__uint128_t string_to_int(string nstring, unsigned int n);
__uint128_t random_128_int(unsigned int k);

// evidence calculation
map<__uint128_t, unsigned int> build_pdata(vector<pair<__uint128_t, unsigned int>> &data, __uint128_t community);
double icc_evidence(__uint128_t community, Partition &p_struct);
double get_evidence(__uint128_t community, Partition &p_struct);

// partitions and data
void get_data(string fname, Partition &p_struct);
void independent_partition(Partition &p_struct);
void random_partition(Partition &p_struct);
void load_partition(Partition &p_struct, string pname);
void parse_community(Partition &p_struct, __uint128_t community, int i);

// proposal functions
void merge_partition(Partition &p_struct);
void split_partition(Partition &p_struct);
void switch_partition(Partition &p_struct);

// main algorithms
void simulated_annealing(Partition &p_struct, 
	unsigned int max_iterations, 
	unsigned int max_no_improve);
void greedy_merging(Partition &p_struct);









