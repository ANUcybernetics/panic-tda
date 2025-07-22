#!/bin/bash

# Check if downsample argument was provided
if [ $# -eq 0 ]; then
    echo "Error: Downsample factor required"
    echo "Usage: $0 <downsample_factor>"
    echo "Example: $0 10  # Process every 10th embedding"
    exit 1
fi

DOWNSAMPLE=$1

# Validate that downsample is a positive integer
if ! [[ "$DOWNSAMPLE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Downsample factor must be a positive integer"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/clustering_${TIMESTAMP}.log"

echo "Starting clustering for all experiments (downsample factor: $DOWNSAMPLE)"
echo "To tail log file: tail -f $LOG_FILE"

# Run the clustering in background with nohup
# Without --force flag to preserve existing clustering results
nohup uv run panic-tda cluster embeddings --downsample $DOWNSAMPLE > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"
echo "Log file: $LOG_FILE"

# Show initial output (first 20 lines)
echo "To follow the log in real-time, run: tail -f $LOG_FILE"
