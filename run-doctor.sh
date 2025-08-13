#!/bin/bash

# Parse arguments
EXPERIMENT_ID=""
OPTIONS=""

# Process all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix|-f)
            OPTIONS="$OPTIONS --fix"
            shift
            ;;
        --yes|-y)
            OPTIONS="$OPTIONS --yes"
            shift
            ;;
        --experiment|-e)
            EXPERIMENT_ID="--experiment $2"
            shift 2
            ;;
        --format)
            OPTIONS="$OPTIONS --format $2"
            shift 2
            ;;
        --db-path|-d)
            OPTIONS="$OPTIONS --db-path $2"
            shift 2
            ;;
        *)
            # If it's a UUID (first non-option argument), treat it as experiment ID
            if [[ -z "$EXPERIMENT_ID" && "$1" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
                EXPERIMENT_ID="--experiment $1"
            else
                echo "Unknown option: $1"
                echo "Usage: $0 [UUID] [--fix] [--yes] [--experiment UUID] [--format text|json] [--db-path PATH]"
                echo ""
                echo "Examples:"
                echo "  $0                           # Check all experiments"
                echo "  $0 --fix                     # Fix issues in all experiments"
                echo "  $0 UUID                      # Check specific experiment"
                echo "  $0 UUID --fix --yes          # Fix specific experiment without prompts"
                echo "  $0 --experiment UUID --fix   # Alternative syntax"
                exit 1
            fi
            shift
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/doctor_${TIMESTAMP}.log"

# Build the command
COMMAND="uv run panic-tda doctor $EXPERIMENT_ID $OPTIONS"

echo "Starting doctor command: $COMMAND"
echo "Output will be logged to: $LOG_FILE"

# Run the doctor command in background with nohup
nohup $COMMAND > "$LOG_FILE" 2>&1 &

# Print the background job PID
echo "Started background job with PID: $!"

tail -f $LOG_FILE
