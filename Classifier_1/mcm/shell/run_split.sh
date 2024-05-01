#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=staging
#SBATCH --time=:10:00
#SBATCH --mail-type=END
#SBATCH --mail-user=paul@hosek.de

 module load 2022
 module load Miniconda3/4.12.0
 source /sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
 conda deactivate
 conda activate mcm

# recompile for current architecture
rm /../../MinCompSpin_SimulatedAnnealing/bin/*
./../../MinCompSpin_SimulatedAnnealing/compile.bat

letters=("A" "B")
python3 run_split.py
#size=1441
#letter=${letters[0]}
#python3 run_split.py --sample_s  $size --split_letter "$letter"

# ## Loop through the sample sizes
# for size in 100 500 1000 2000 3000; do

# #    # Loop through the letters
#     for letter in "${letters[@]}"; do
# #        # Call sbatch and run run_split.sh with the current sample size and letter
#       python3 run_split.py --sample_s  $size --split_letter "$letter"
#     done
# done