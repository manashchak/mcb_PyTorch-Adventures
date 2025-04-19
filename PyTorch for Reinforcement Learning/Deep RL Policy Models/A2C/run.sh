#!/bin/bash

# List of lambda values to test
NENVS=(1 2 4 8 16)

# Maximum number of concurrent processes
MAX_CONCURRENT=5

# Counter for running processes
running=0

for N in "${NENVS[@]}"; do
    
    # Run the Python script in the background
    echo "Starting experiment with NENVs=$N"
    python run.py --num_envs "$N" &
    
    # Increment running process counter
    ((running++))
    
    # If maximum concurrent processes reached, wait for one to finish
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        ((running--))
    fi
done

# Wait for all remaining processes to finish
wait

echo "All experiments completed"