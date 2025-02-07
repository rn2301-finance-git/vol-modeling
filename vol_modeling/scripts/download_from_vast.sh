#!/bin/bash

# Check if arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <instance-alias> <remote-path> <local-path>"
    echo "Examples:"
    echo "  $0 vast1 /workspace/vol_project/logs/training.log ./logs/    # Download single file"
    echo "  $0 vast1 /workspace/vol_project/models/best.pth ./models/    # Download model checkpoint"
    exit 1
fi

# Source the aliases script to get access to functions
source "$(dirname "$0")/aliases.sh"

# Arguments
INSTANCE_ALIAS=$1
REMOTE_PATH=$2
LOCAL_PATH=$3

# Get instance details
INSTANCE_DETAILS=$(get_instance_details "$INSTANCE_ALIAS")

if [ "$INSTANCE_DETAILS" = "null" ]; then
    echo "Error: Instance alias '$INSTANCE_ALIAS' not found"
    echo "Use './aliases.sh list' to see available instances"
    exit 1
fi

# Extract instance details
VAST_IP=$(echo "$INSTANCE_DETAILS" | jq -r '.host')
VAST_PORT=$(echo "$INSTANCE_DETAILS" | jq -r '.port')
VAST_USER=$(echo "$INSTANCE_DETAILS" | jq -r '.user')
VAST_KEY=$(echo "$INSTANCE_DETAILS" | jq -r '.auth_file')

# Rsync command
echo "Downloading from $VAST_USER@$VAST_IP:$REMOTE_PATH to $LOCAL_PATH ..."
rsync -avz -e "ssh -i $VAST_KEY -p $VAST_PORT" \
    "$VAST_USER@$VAST_IP:$REMOTE_PATH" \
    "$LOCAL_PATH"

echo "Download complete!"
