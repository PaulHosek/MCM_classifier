## Community detection using Minimally Complex Models: Simulated annealing algorithm

This program contains two algorithms, greedy merging (GM) and simulated annealing (SA), for finding community structures in binary data of up to 128 variables by using inference based on a class of spin models (maximum entropy models for binary data) called Minimally Complex Models (MCM). 
### Simulated annealing
The SA algorithm works by calculating the log-evidence $\log E$ for a given initial partition $C_i$ (community structure). It then proposes slight changes to the partition and calculates the difference in log-evidence $\Delta \log E$. 
If the new partition $C_{i+1}$ has a larger log-evidence $\Delta \log E > 0$, the new partition is accepted. If not, the partition is still accepted with probability $P(C_{i+1}=C_i)\sim \exp(\Delta \log E/T_A)$ where $T_A$ is the annealing temperature. This parameter controls how likely the new partition is accepted. At large values, this prevents getting stuck in locally optimal solutions. 
During the search the annealing temperature is gradually lowered allowing the algorithm to converge to an optimal solution. This is done according to a logarithmic cooldown schedule where the annealing temperature at iteration $i$ is given by $T_A(i)=T_0/(1+\log(1+i))$.
See https://en.wikipedia.org/wiki/Simulated_annealing for more details.

The algorithm can change the partition in three ways:
- merge: two communities are merged into a single community
- split: a single community is split into two communities (not necessarily of the same size)
- switch: a node from one community is placed inside another community
### Greedy merging
The GM algorithm starts from an independent partition where each node is assigned to a separate community. It then checks all possible mergers of two communities and accept the merger that increases the log-evidence by the largest amount. The algorithm stops when there is no possible increase in log-evidence or when all nodes have been assigned to the same community. This algorithm is much faster than the SA algorithm but may not converge to the optimal solution. A useful approach is therefore to use the best partition found with the GM algorithm as the initial condition for the SA algorithm (see "How to use").

## Requirements
The code uses C++ version 11.

## Installation

### Windows

The code can be compiled using the `compile.bat` batch file in the main folder. This creates an `saa.exe` executable file in the `./bin` folder. Alternatively, the code can be compiled using the command `g++ -std=c++11 -O3 -Wall ./src/*.cpp -o ./bin/saa.exe`. There is an additional batch file `compile_and_run.bat` which is useful for testing purposes. It compiles the code and then runs an analysis. This file also shows the use of the optional flags `--max` and `--stop`.

### Linux / macOS

The code can be compiled using the command `g++ -std=c++11 -O3 -Wall ./src/*.cpp -o ./bin/saa.out`.

## Tutorial

A tutorial in the form of a jupyter notebook has been provided in the `./doc/` directory.

## Running the code

The code is run from the executable file `saa.exe` (on Windows) or `saa.out` (on Linux/Mac) in the `bin` folder. The program should be called with the number of variables `n` and the filename of the dataset as required arguments. The data is assumed to be in the `./input/data` folder and should end with `.dat` extension. The data should be encoded as binary strings, e.g. `11001110010` which are read from right to left. The maximum number of variables is 128.

- To see an example, run the `run.bat` batch file from the main folder (on Windows).

### How to use:

The `./input/data` folder contains several example datasets on 20, 40, 60, 80 and 100 variables. The `./input/comms` folder contains initial partitions for 20 and 100 variables.

To analyse the dataset `./input/data/DATAFILE_NAME.dat` run the command (for Linux/Mac, replace .exe with .out):

`saa.exe n -i DATAFILE_NAME`

where `n` is the number of variables and the flag `-i` should be followed by the name of the dataset (without the `.dat` extension).

In this case, the initial partition is a random partition. Optionally, to load an initial partition, use:

`saa.exe n -i DATAFILE_NAME -p PARTITION_NAME`

This partition should be located in the `./input/comms` folder and have the name `PARTITION_NAME.dat`. By default, the program runs for 50k iterations and stops early if there has been no improvement in the log-evidence for 10k iterations. These parameters can be changed by using the `--max` and `--stop` flags. For example, to perform a maximum of 4500 iterations and stop early after 2250 "unsuccessful" iterations, use:

`saa.exe n -i DATAFILE_NAME --max 4500 --stop 2250`

By default, the program runs only the simulated annealing algorithm and starts from an independent partition. To run a single pass of the greedy merging algorithm before SA, use the `-g` flag. For example, the following would run both the GM and SA algorithms

`saa.exe n -i DATAFILE_NAME -g`

To run greedy merging only, use the `-s` flag together with the `-g` flag. Note that using only the `-s` flag will disable both algorithms and the program will just return the initial partition.

`saa.exe n -i DATAFILE_NAME -g -s`

Finally, to start from a random partition instead of the independent partition, use the `-r` flag. 

`saa.exe n -i DATAFILE_NAME -r`

## Output 

The best partition found is written to `./output/comms/DATAFILE_NAME_comms.dat`. The associated best log-evidence is written to `./output/stats/DATAFILE_NAME_stats.dat`. 

### Example: 

If the best partition found divides 20 variables in the following way: `[[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12,13,14],[15,16,17,18,19]]` the output file in the `./output/comms` directory would like this:

```
00000000000000000111
00000000000000111000
00000000000111000000
00000000011000000000
00000111100000000000
11111000000000000000
```

## Important parameters

- The parameter `EPSILON` in `./src/header.h` controls the minimum difference in log-evidence the best partition should have from the previous best in ordered to be considered the new best. Setting this to 0 can result in partitions that are equivalent up to permutation to be considered better due to tiny numerical differences. This can cause the algorithm to perform more iterations than strictly necessary.
- There are four parameters that control the simulated annealing procedure in `./src/main.cpp`. The current settings perform well as a good starting point that balances performance and optimality of the found partition. 
  - `T0`: the initial annealing temperature.
  - `update_schedule`: the number of iterations the algorithms performs at the same annealing temperature.
  - `max_no_improve`: the maximum number of iterations without improvement in log-evidence before stopping the algorithm - increasing this allows for a more exhaustive search at the cost of speed and can be useful when analyzing data of many variables.
  - `max_iterations`: the maximum total number of iterations to perform.
- The last two parameters can be specified at runtime using the `--max` and `--stop` flags.

## Coming soon
- Python wrapper

  

