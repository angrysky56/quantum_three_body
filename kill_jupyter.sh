#!/bin/bash
# Quick script to kill all Jupyter processes

echo "Finding and killing Jupyter processes..."

# For Linux - Find all jupyter notebook processes
pids=$(ps aux | grep "[j]upyter-notebook\|[j]upyter notebook" | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No Jupyter processes found."
else
    echo "Found Jupyter processes with PIDs: $pids"
    for pid in $pids; do
        echo "Killing process $pid..."
        kill -9 $pid
    done
    echo "Done."
fi

# Also check ports
for port in 8888 8889 8890 8891 8892; do
    pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo "Found process using port $port with PID: $pid"
        echo "Killing process $pid..."
        kill -9 $pid
    fi
done

echo "All Jupyter processes should be terminated."
echo "You can now start a fresh Jupyter notebook instance."
