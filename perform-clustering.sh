#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/clustering_${TIMESTAMP}.log"

echo "Starting clustering for all experiments (no downsampling)"
echo "To tail log file: tail -f $LOG_FILE"

# Run the clustering in background with nohup
# Using --force to re-cluster any existing results
nohup uv run panic-tda cluster-embeddings --force > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"
echo "Log file: $LOG_FILE"

# Show initial output (first 20 lines)
echo "To follow the log in real-time, run: tail -f $LOG_FILE"