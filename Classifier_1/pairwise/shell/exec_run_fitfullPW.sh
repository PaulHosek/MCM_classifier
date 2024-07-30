#!/bin/bash



# for run in {0..9}
# do

#for size in 10 100 500 $(seq 1000 1000 6000) 6315 ; do 

# for 6315 in  500;do 
#   echo $size
#    for digit in 0,1,3,5; do 
#    do
#     python3 run_fitfullPW.py --sample_s $size --digit $digit --method "ace" --trainfull "full"
#    done
# done


# done

for size in 6315; do 
  echo $size
  for digit in 1 3 5; do 
    timeout 14400 python3 run_fitfullPW.py --sample_s $size --digit $digit --method "ace" --trainfull "full"
  done
done
