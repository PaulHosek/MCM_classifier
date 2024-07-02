#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <bitset>
#include <time.h>
#include <sstream>
#include <string>
#include <map>

using namespace std;

const int n = 121;
const int N = 11111;

stringstream fname_base;

vector<vector<double> > random_jij(int n);
void write_jij(vector<vector<double> > jij);
void generate_data(vector<vector<double> > jij, unsigned int N);

int main() {

	fname_base << "../data/test_data_n" << n << "_N" << N;

	vector<vector<double> > jij = random_jij(n);
	write_jij(jij);
	generate_data(jij, N);


	return 0;

}

vector<vector<double> > random_jij(int n) {

	// generate random jij in [-1,1]

	vector<vector<double> > jij(n, vector<double>(n));

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			jij[i][j] = -1 + static_cast <double> (rand()) / static_cast <double> (RAND_MAX/2);
			jij[j][i] = jij[i][j];

			cout << "j" << (i+1) << "," << (j+1) << ": " << jij[i][j] << endl;

		}
	}

	return jij;

}


void write_jij(vector<vector<double> > jij) {

	/* write jij to file using same 
	convention as inference output */

	uint64_t op;
	ofstream myfile;

	string fname = fname_base.str() + "_jij.dat";

	map<uint64_t, double> jij_map;
	map<uint64_t, double>::iterator it;

	myfile.open(fname);

	// convert jij to map to ensure same ordering in output files

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {

			op = pow(2,i) + pow(2,j);
			jij_map[op] = jij[i][j];

		}
	}

	
	for (it = jij_map.begin(); it != jij_map.end(); it++) {

		myfile << bitset<n>(it->first) << "\t" << it->second << "\n";

	}

	myfile.close();

}

void generate_data(vector<vector<double> > jij, unsigned int N) {

	srand(time(NULL));

	int steps = N * n;
	int therm = N;
	int i;
	double energy;
	double delta_e;
	double u;

	int state[n];

	for (int i = 0; i < n; i++) {
		state[i] = -1 + 2 * (rand() % 2);
	}

	ofstream data;
	string fname = fname_base.str() + ".dat";
	data.open(fname);

	for (int step = 0; step < steps; step++){

		i = rand() % n;

		energy = 0;

		for (int j = 0; j < n; j++) {

			energy += jij[i][j] * state[n-1-j];
		}

		delta_e = 2 * state[n-1-i] * energy;

		if (delta_e <= 0) {

			state[n-1-i] *= -1;

		} else {

			u = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

			if (exp(-delta_e) > u) {

				state[n-1-i] *= -1;
			}
		}

		if (step % (n) == 0 && step >= therm) {

			for (int i = 0; i < n; i++) {

				data << (state[i]+1)/2;
			}

			data << "\n";


		}



	}

	data.close();


}