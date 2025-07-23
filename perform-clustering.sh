#!/bin/bash

# Check if both arguments were provided
if [ $# -ne 2 ]; then
    echo "Error: Both downsample factor and epsilon are required"
    echo "Usage: $0 <downsample_factor> <epsilon>"
    echo "Example: $0 10 0.5  # Process every 10th embedding with epsilon=0.5"
    exit 1
fi

DOWNSAMPLE=$1
EPSILON=$2

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/clustering_${TIMESTAMP}.log"

echo "Starting clustering for all experiments (downsample factor: $DOWNSAMPLE, epsilon: $EPSILON)"
echo "To tail log file: tail -f $LOG_FILE"

# Run the clustering in background with nohup
# Without --force flag to preserve existing clustering results
nohup uv run panic-tda cluster embeddings --downsample $DOWNSAMPLE --epsilon $EPSILON > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"
echo "Log file: $LOG_FILE"

# Show initial output (first 20 lines)
echo "To follow the log in real-time, run: tail -f $LOG_FILE"
