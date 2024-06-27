#!/bin/bash

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output-path) output_path="$2"; shift ;;
        --concurrency-range) concurrency_range="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default values if not provided
output_path=${output_path:-"./"}
concurrency_range=${concurrency_range:-"1:2:1"}

log_filename=$output_path/server.log
server_pid_file=$output_path/server.pid

# Ensure the output directory exists
mkdir -p "$output_path"

# Set a trap to ensure tritonserver is always killed on exit
trap 'echo "Killing triton server..."; if [[ -f "$server_pid_file" ]]; then kill "$(cat "$server_pid_file")"; fi' EXIT

# Launch the server
nohup tritonserver --model-repository=/workspace/backend/ > ${log_filename} 2>&1 &
server_pid=$!
echo $server_pid > $server_pid_file

# Wait for the server to start
sleep 60

# Check server's readiness
check_server_ready() {
    local max_retries=10
    local retry_interval=20  # wait 20 seconds before re-trying
    local retry_count=0
    local server_ready=0  # 0 means not ready, 1 means ready

    echo "Checking if server is ready..."

    while [[ $retry_count -lt $max_retries && $server_ready -eq 0 ]]; do
        # Use curl to check the server's status. The -s flag silences curl's output, and -o /dev/null discards the actual content.
        local response=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/ready)
        
        if [[ "$response" == "200" ]]; then
            server_ready=1
            echo "Server is ready!"
            echo ""
        else
            echo "Server not ready, retrying in $retry_interval seconds..."
            echo ""
            retry_count=$((retry_count + 1))
            sleep $retry_interval
        fi
    done

    if [[ $server_ready -eq 0 ]]; then
        echo "Server didn't become ready after $max_retries attempts. Exiting..."
        exit 1
    fi
}

check_server_ready

# Call the performance analyzer via call_perf_analyzer.py 
python /workspace/evaluation/call_perf_analyzer.py --output_path "$output_path" --concurrency_range "$concurrency_range"

# Kill the server
echo "Killing triton server..."
kill $server_pid

