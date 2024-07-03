#include <iostream>
#include <string>
#include <fstream>
#include <map>

#include <Eigen/Core>
#include <LBFGS.h>

using namespace std;
using namespace Eigen;
using namespace LBFGSpp;

const __uint128_t ONE = 1;

// function declarations
map<__uint128_t, unsigned int> read_data(string fname, unsigned int &N, unsigned int n, string directory);
map<__uint128_t, double> get_pdata(map<__uint128_t, unsigned int> &Nset, unsigned int N);
map<__uint128_t, double> optimize(unsigned int n, map<__uint128_t, double> &pdata);
void write_jij(map<__uint128_t, double> &jij, string fname, unsigned int n, string directory);

// MAIN FUNCTION
int main(int argc, char **argv) {

	unsigned int n = 1;
	unsigned int N = 0;
	string directory;
	string fname;
	
	for (int i = 1; i < argc; ++i) {
	if (std::string(argv[i]) == "-n" && i + 1 < argc) { // Check for -n flag and ensure there's a value following it
		n = std::stoi(argv[++i]); // Convert the next argument to integer and assign to N
		std::cout << "-n set to: " << n << "\n";
	} else if (std::string(argv[i]) == "-i" && i + 1 < argc) { // Check for -i flag and ensure there's a value following it
		fname = argv[++i]; // Assign the next argument to fname
		std::cout << "- Input file: " << fname << "\n";
	} else if (std::string(argv[i]) == "-p" && i + 1 < argc) { // Check for -d flag and ensure there's a value following it
		directory = std::string("./") + argv[++i] + "/"; // Assign the next argument to directory
		std::cout << "- Path: " << directory << "\n";
	}
    }
    // sscanf(argv[1], "%d", &n);
    // for (int i = 2; i < argc; i++) {

    //     string arg = argv[i];

    //     // input data
    //     if (arg == "-i") {
    //         fname = argv[i+1];
    //         i++;
    //         cout << "- input file: " << fname << "\n";
    //     }
    // }



	map<__uint128_t, unsigned int> Nset = read_data(fname, N, n, directory);
	map<__uint128_t, double> pdata = get_pdata(Nset, N);
	map<__uint128_t, double> jij = optimize(n, pdata);
	write_jij(jij, fname, n, directory);
	return 0;
}

// HELPER FUNCTIONS =================================================
string int_to_bitstring(__uint128_t number, unsigned int r) {

	// https://leetcode.com/problems/add-binary/solutions/143963/c-2ms-solution-using-128-bit-integers/

	string s; 

	while (number) {

		s.push_back((number & ONE) + '0');
		number >>= 1;
	}

	reverse(s.begin(), s.end()); // reverse string
	s = string(r - s.length(), '0').append(s); // append leading zeros

	return s;

}

unsigned int bit_count(__uint128_t number) {

	unsigned int count;

	for (count = 0; number; count++) {
		number &= (number - 1);
	}

	return count;
}

__uint128_t string_to_int(string nstring, unsigned int n) {

	__uint128_t state, op;
	char c = '1';

	op = ONE << (n - 1);
	state = 0;

	for (auto &elem: nstring) {
			if (elem == c) {state += op;}
			op = op >> 1;
		}

	return state;
}

// PROCESSING FUNCTIONS ============================================
map<__uint128_t, unsigned int> read_data(string fname, unsigned int &N, unsigned int n, string directory) {

	cout << "reading data..." << endl;

	string line, subline;
	__uint128_t state;

	map<__uint128_t, unsigned int> Nset;

	string fname_full = directory + fname + ".dat";

	cout << fname << endl;
	ifstream myfile(fname_full);

	while (getline(myfile, line)) {

		subline = line.substr(0, n);
		state = string_to_int(subline, n);
		Nset[state] += 1;
		N++;

	}

	myfile.close();

	return Nset;
}

map<__uint128_t, double> get_pdata(map<__uint128_t, unsigned int> &Nset, unsigned int N) {

	map<__uint128_t, double> pdata;
	map<__uint128_t, unsigned int>::iterator it;

	for (it = Nset.begin(); it != Nset.end(); it++) {

		pdata[it->first] = it->second / (double) N;

	}

	return pdata;

}

// modifided function to fit my convention: 0s for the fields before and only the couplings not the bitstrings
void write_jij(map<__uint128_t, double> &jij, string fname, unsigned int n, string directory) {

	map<__uint128_t, double>::iterator it;

	ofstream myfile;

	string fname_full = directory + fname + "_rise.dat";

	myfile.open(fname_full);

	// write 
	for (int i=0; i<n; i++){
		myfile << 0.0 << "\n";
	}

	for (it = jij.begin(); it != jij.end(); it++) {

		// myfile << int_to_bitstring(it->first, n) << "\t" << it->second << "\n";
		myfile << it->second << "\n";
	}

	myfile.close();

}

// MAIN ALGORITHM CLASS
class rise_obj_func {
private:
	map<__uint128_t, double> pdata;
	unsigned int node;
	unsigned int n;
public:
	rise_obj_func(map<__uint128_t, double> pdata_, unsigned int node_, unsigned int n_) : pdata(pdata_), node(node_), n(n_) {}

	double operator()(const VectorXd &x, VectorXd &grad) {

		double sn = 0;
		double energy, p;

		__uint128_t state, op_and;
		unsigned int op_bits;

		int eval_op;
		int eval_ops[n];

		map<__uint128_t, double>::iterator it;

		double g[n];
		for (unsigned int i = 0; i < n; i++) {
			g[i]= x[i];
			grad[i] = 0;
		}

		g[node] = 0;

		for (it = pdata.begin(); it != pdata.end(); it++) {

			state = it->first;
			p = it->second;

			energy = 0;

			for (unsigned int j = 0; j < n; j++) {

				if (j == node) {continue;}

				__uint128_t op = 0;
				__uint128_t s1 = 0;
				__uint128_t s2 = 0;

				s1 = (ONE << node);
				s2 = (ONE << j);

				op = s1 + s2;

				op_and = (op & state);
				op_bits = bit_count(op_and);
				op_bits %= 2;
				eval_op = 1 - 2 * op_bits;
				energy += g[j] * eval_op;

				eval_ops[j] = eval_op;

			}

			sn += exp(-energy) * p;

			for (unsigned int j = 0; j < n; j++) {

				grad[j] += (-eval_ops[j] * exp(-energy)) * p;
			}
		}

		grad[node] = 0;

		return sn;

	}
};

// OPTIMIZER FUNCTION
map<__uint128_t, double> optimize(unsigned int n, map<__uint128_t, double> &pdata) {

	LBFGSParam<double> param;
	param.epsilon = 1e-4;
	param.max_iterations = 10000;

	LBFGSSolver<double> solver(param);

	map<__uint128_t, double> jij;
	__uint128_t ONE = 1;

	for (unsigned int node = 0; node < n; node++) {

		rise_obj_func rise(pdata, node, n);
		VectorXd g = VectorXd::Zero(n);
		double min_f;
		int niter = solver.minimize(rise, g, min_f);

		cout << "node: " << node << "\t";
		cout << "f: " << min_f << "\t";
		cout << "iterations: " << niter << endl;

		for (unsigned int j = 0; j < n; j++) {

			if (j == node) {continue;}

			__uint128_t op = 0;
			__uint128_t s1 = 0;
			__uint128_t s2 = 0;

			s1 = (ONE << node);
			s2 = (ONE << j);
			op = s1 + s2;

			jij[op] += g[j]/2;
		}

	}

	return jij;

}


