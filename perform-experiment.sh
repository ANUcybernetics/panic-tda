#!/bin/bash

# Check if config file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <config_file>"
  exit 1
fi

CONFIG_FILE=$1

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

echo "Starting experiment with config: $CONFIG_FILE"
echo "To tail log file: tail -f $LOG_FILE"

# Run the experiment in background with nohup
nohup uv run panic-tda experiment perform "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"

tail -f $LOG_FILE
