#include "header.h"

void get_data(string fname, Partition &p_struct) {

	// https://github.com/clelidm/MinCompSpin_Greedy

	__uint128_t state;
	map<__uint128_t, unsigned int> data;

	string fpath = "../input/data/" + fname + ".dat";
	string line, subline;
	ifstream myfile(fpath);

	while (getline(myfile, line)) {

		subline = line.substr(0, p_struct.n);
		state = string_to_int(subline, p_struct.n);
		data[state]++;
		p_struct.N++;

	}

	myfile.close();

	cout << "- loaded: " << fpath;
	cout << " (" << p_struct.N << " samples)\n" << endl;

	// convert map to vector
	for (auto &my_pair : data) {
		p_struct.data.push_back(my_pair);
	}

}