#!/bin/bash

# List of lambda values to test
LAMBDAS=(0.01 0.3 0.7 0.9 0.95 0.99)

# Maximum number of concurrent processes
MAX_CONCURRENT=8

# Counter for running processes
running=0

for lambda in "${LAMBDAS[@]}"; do
    
    # Run the Python script in the background
    echo "Starting experiment with lambda=$lambda, logging to $LOG_FILE"
    python run.py --lambda_weight "$lambda" &
    
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

echo "All experiments completed. Logs are in $OUTPUT_DIR"