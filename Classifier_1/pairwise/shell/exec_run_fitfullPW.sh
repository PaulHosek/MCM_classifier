#!/bin/bash



for digit in `seq 0 2` # inclusive on both ends
do
    python3 run_fitfullPW.py --sample_s 10 --digit $digit
done
