#!/bin/bash



# for run in {0..9}
# do

#for size in 10 100 500 $(seq 1000 1000 6000) 6315 ; do 
for size in  6000;do 
  echo $size
   # for digit in {0..9} 
   # do
    python3 run_fitfullPW.py --sample_s $size --digit 0 --method "rise"
   # done
done

# done

