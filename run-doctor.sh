#!/bin/bash

# Check if UUID is provided
if [ -z "$1" ]; then
    echo "Error: UUID argument is required"
    echo "Usage: $0 <uuid>"
    exit 1
fi

UUID="$1"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/doctor_${TIMESTAMP}.log"

echo "Starting doctor --fix with UUID: $UUID"
echo "Output will be logged to: $LOG_FILE"

# Run the doctor --fix command in background with nohup
nohup uv run panic-tda doctor --fix $UUID > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"

tail -f $LOG_FILE
