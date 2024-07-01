#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <bitset>
//#include <sstream>

using namespace std;

#include <Eigen/Core>
#include <LBFGS.h>

using namespace Eigen;
using namespace LBFGSpp;

const int n = 60;
const unsigned int N = 1000000;


const string fname = "../data/test_data_n60_N1000000";
//stringstream fname_base;

// function declarations
map<__uint128_t, unsigned int> read_data();
map<__uint128_t, double> get_pdata(map<__uint128_t, unsigned int> &Nset, unsigned int N);
map<__uint128_t, double> optimize(map<__uint128_t, double> &pdata);
void write_jij(map<__uint128_t, double> &jij);

int main() {

	//fname_base << "../data/test_data_n" << n << "_N" << N;

	map<__uint128_t, unsigned int> Nset = read_data();
	map<__uint128_t, double> pdata = get_pdata(Nset, N);
	map<__uint128_t, double> jij = optimize(pdata);
	write_jij(jij);

	return 0;

}

map<__uint128_t, unsigned int> read_data() {

	cout << "reading data..." << endl;

	string line, subline;
	__uint128_t state;

	map<__uint128_t, unsigned int> Nset;

	//string fname = fname_base.str() + ".dat";

	string fname_full = fname + ".dat";

	cout << fname << endl;
	ifstream myfile(fname_full);

	while (getline(myfile, line)) {

		subline = line.substr(0,n);
		state = bitset<n>(subline).to_ulong();
		Nset[state] += 1;

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

void write_jij(map<__uint128_t, double> &jij) {

	map<__uint128_t, double>::iterator it;

	ofstream myfile;

	string fname_full = fname + "_jij_fit.dat";

	myfile.open(fname_full);

	for (it = jij.begin(); it != jij.end(); it++) {

		myfile << bitset<n>(it->first) << "\t" << it->second << "\n";

	}

	myfile.close();

}


class rise_obj_func {
private:
	map<__uint128_t, double> pdata;
	int node;
public:
	rise_obj_func(map<__uint128_t, double> pdata_, int node_) : pdata(pdata_), node(node_) {}

	double operator()(const VectorXd &x, VectorXd &grad) {

		double sn = 0;
		double energy, p;

		__uint128_t state, op, op_and;
		unsigned int op_bits;

		int eval_op;
		int eval_ops[n];

		map<__uint128_t, double>::iterator it;

		double g[n];
		for (int i = 0; i < n; i++) {
			g[i]= x[i];
			grad[i] = 0;
		}

		g[node] = 0;

		for (it = pdata.begin(); it != pdata.end(); it++) {

			state = it->first;
			p = it->second;

			energy = 0;

			for (int j = 0; j < n; j++) {

				op = pow(2,node) + pow(2,j);
				op_and = (op & state);
				op_bits = bitset<n>(op_and).count();
				op_bits %= 2;
				eval_op = 1 - 2 * op_bits;
				energy += g[j] * eval_op;

				eval_ops[j] = eval_op;

			}

			sn += exp(-energy) * p;

			for (int j = 0; j < n; j++) {

				grad[j] += (-eval_ops[j] * exp(-energy)) * p;
			}
		}

		grad[node] = 0;

		return sn;

	}
};

map<__uint128_t, double> optimize(map<__uint128_t, double> &pdata) {

	LBFGSParam<double> param;
	param.epsilon = 1e-4;
	param.max_iterations = 1000;

	LBFGSSolver<double> solver(param);

	map<__uint128_t, double> jij;
	__uint128_t ONE = 1;

	for (int node = 0; node < n; node++) {

		rise_obj_func rise(pdata, node);
		VectorXd g = VectorXd::Zero(n);
		double min_f;
		int niter = solver.minimize(rise, g, min_f);

		cout << "node: " << node << "\t";
		cout << "f: " << min_f << "\t";
		cout << "iterations: " << niter << endl;

		for (int j = node + 1; j < n; j++) {

			__uint128_t op = 0;
			__uint128_t s1 = 0;
			__uint128_t s2 = 0;

			s1 = (ONE << node);
			s2 = (ONE << j);

			op = s1 + s2;

			// cout << bitset<n>(s1) << endl;
			// cout << bitset<n>(s2) << endl;
			// cout << bitset<n>(op) << endl;

			jij[op] = g[j];
		}

	}

	return jij;

}