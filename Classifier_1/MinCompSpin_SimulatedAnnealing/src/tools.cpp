#include "header.h"


bool DoubleSame(double a, double b){
	return fabs(a-b) < EPSILON;
}

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

__uint128_t random_128_int(unsigned int k){

	// generate random 128-bit number less then 2^k

	__uint128_t number = 0;

	number += rand() % 2;

	for (unsigned i = 1; i < k; i++){	
		number <<= 1;
		number += rand() % 2;
	}

	return number;
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

unsigned int bit_count(__uint128_t number) {

	unsigned int count;

	for (count = 0; number; count++) {
		number &= (number - 1);
	}

	return count;
}

