#include "header.h"


unsigned int randomBitIndex(__uint128_t number) {

	string r;

	unsigned int nb = bit_count(number);

	unsigned int x = 0;
	unsigned int y = 0;
	
	unsigned int my_index = rand()/(RAND_MAX/nb);
	
	unsigned int loc = 0;

	while(number) {

		r = ((number & ONE) + '0');

		if (r == "1") {
			if (x == my_index){
				loc = y;
				break;
			}
			x++;
		}

		number >>= 1;
		y++;

	}	

	return loc;

}