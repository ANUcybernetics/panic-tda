#!/bin/bash

# Check if at least one config file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <config_file1> [config_file2] [config_file3] ..."
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

echo "Starting experiments with configs: $@"
echo "To tail log file: tail -f $LOG_FILE"

# Start all experiments under a single nohup process
(
  for CONFIG_FILE in "$@"; do
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Starting experiment with config: $CONFIG_FILE"
    uv run trajectory-tracer perform-experiment "$CONFIG_FILE"
    echo "$(date +"%Y-%m-%d %H:%M:%S") - Finished experiment with config: $CONFIG_FILE"
    echo "-------------------------------------------"
  done
) > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"

tail -f $LOG_FILE
