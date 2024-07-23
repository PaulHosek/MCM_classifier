#!/bin/bash

# Number of iterations

iterations=50 

timeout=360

# for size in 3000 4000 5000 6315; do # 10 100 2000 3000 4000 5000 6315;
#for size in 3000 4000 5000 6315; do # 10 100 2000 3000 4000 5000 6315;
for size in $(seq 500 500 6000) 10 100 200 300 400 600 700 800 900 ./ 6315; do
  for (( i=1; i <= iterations; i++ )); do  # Removed $ from iterations for consistency
    
    # Run the Python script with timeout in the background
    timeout -s SIGTERM $timeout python3 run_fullsample.py --sample_s $size 
    python_pid=$!

    # Wait for the Python script to finish or timeout
    wait $python_pid

    # Check the exit status and kill process if necessary
    if [[ $? -ne 0 ]]; then
      echo "Python script timed out or crashed. Killing process..."
      kill -SIGTERM $python_pid
    fi

    # Add a short delay between iterations (optional)
    sleep 5
  done
done


