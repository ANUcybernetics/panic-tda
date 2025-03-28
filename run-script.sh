#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/script_${TIMESTAMP}.log"

echo "Starting script with config: $CONFIG_FILE"
echo "Output will be logged to: $LOG_FILE"

# Run the script in background with nohup
nohup uv run trajectory-tracer script > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"
