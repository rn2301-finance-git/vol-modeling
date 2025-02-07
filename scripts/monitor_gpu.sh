#!/bin/bash

CONFIG_DIR="$HOME/.bam_config"
INSTANCES_FILE="$CONFIG_DIR/instances.json"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <instance-alias>"
    echo "Use ./aliases.sh list to see available instances"
    exit 1
fi

ALIAS=$1
INSTANCE_INFO=$(jq -r --arg alias "$ALIAS" '.[$alias]' "$INSTANCES_FILE")

if [ "$INSTANCE_INFO" = "null" ]; then
    echo "Instance alias not found. Use ./aliases.sh list to see available instances"
    exit 1
fi

# Parse instance info
HOST=$(echo "$INSTANCE_INFO" | jq -r '.host')
PORT=$(echo "$INSTANCE_INFO" | jq -r '.port')
USER=$(echo "$INSTANCE_INFO" | jq -r '.user')
AUTH_TYPE=$(echo "$INSTANCE_INFO" | jq -r '.auth_type')
AUTH_FILE=$(echo "$INSTANCE_INFO" | jq -r '.auth_file')
AUTH_FILE=$(eval echo "$AUTH_FILE")  # Expand ~ if present

# Build SSH command based on auth type
if [ "$AUTH_TYPE" = "key" ]; then
    SSH_CMD="ssh -i \"$AUTH_FILE\" -p $PORT"
else
    SSH_CMD="ssh -p $PORT"
fi

# Function to run SSH commands with retries
run_ssh() {
    local max_attempts=3
    local attempt=1
    local delay=2

    while [ $attempt -le $max_attempts ]; do
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$USER@$HOST" "$1" && return 0
        
        echo "Connection attempt $attempt failed. Retrying in $delay seconds..."
        sleep $delay
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done

    echo "❌ Could not connect to $HOST after $max_attempts attempts"
    return 1
}

while true; do
    clear
    echo "System Monitor for $HOST"
    echo "----------------------------------------"
    
    # Test connection first
    if ! run_ssh "echo 'Connection test'" > /dev/null 2>&1; then
        echo "⚠️  Unable to connect to host. Retrying in 10 seconds..."
        sleep 10
        continue
    fi
    
    # GPU Status
    echo -e "\n=== GPU Status ==="
    run_ssh "nvidia-smi" || echo 'nvidia-smi not available'
    
    # CPU Usage
    echo -e "\n=== CPU Usage ==="
    run_ssh '
        echo "Load averages - 1, 5, 15 min:"
        cat /proc/loadavg 2>/dev/null || uptime 2>/dev/null || echo "CPU info not available"
        echo ""
        echo "Top processes by CPU:"
        ps aux --sort=-%cpu | head -n 4 2>/dev/null || echo "Process info not available"
    '
    
    # Memory Usage
    echo -e "\n=== Memory Usage ==="
    run_ssh '
        echo "Top processes by memory usage:"
        ps aux --sort=-%mem | head -n 4 2>/dev/null || echo "Memory info not available"
    '
    
    # Disk Usage
    echo -e "\n=== Disk Usage ==="
    run_ssh 'df -h / || echo "Disk usage info not available"'
    
    # System Load
    echo -e "\n=== System Load ==="
    run_ssh 'uptime || echo "System load info not available"'
    
    # Summary and Warnings
    echo -e "\n=== System Health Summary ==="
    run_ssh '
        has_warnings=false
        
        # Check CPU load
        load=$(cat /proc/loadavg 2>/dev/null | cut -d" " -f1)
        if [ -n "$load" ]; then
            cores=$(nproc 2>/dev/null || echo 1)
            if [ -n "$cores" ]; then
                load_per_core=$(echo "scale=2; $load/$cores" | bc 2>/dev/null)
                if [ -n "$load_per_core" ]; then
                    if echo "$load_per_core > 2.0" | bc -l 2>/dev/null | grep -q 1; then
                        echo "⚠️  HIGH CPU LOAD: Load per core is $load_per_core"
                        has_warnings=true
                    fi
                fi
            fi
        fi
        
        # Check disk space
        disk_used=$(df / 2>/dev/null | tail -n 1 | awk "{print \$5}" | sed "s/%//" 2>/dev/null)
        if [ -n "$disk_used" ]; then
            if echo "$disk_used > 90" | bc -l 2>/dev/null | grep -q 1; then
                echo "⚠️  LOW DISK SPACE: ${disk_used}% used"
                has_warnings=true
            fi
        fi
        
        # Show all clear if no warnings
        if ! $has_warnings; then
            echo "✅ All monitored systems within normal ranges"
        fi
    '
    
    sleep 5
done