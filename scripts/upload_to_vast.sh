#!/bin/bash

# Check if arguments are provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <instance-alias>"
    echo "Will sync /Users/raghuvar/Code/BAM/ to the instance's configured remote directory/BAM"
    exit 1
fi

# Source the aliases script to get access to functions
source "$(dirname "$0")/aliases.sh"

# Arguments
INSTANCE_ALIAS=$1
LOCAL_PATH="/Users/raghuvar/Code/BAM/"

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
REMOTE_PATH=$(echo "$INSTANCE_DETAILS" | jq -r '.remote_dir')/BAM

# Rsync command
echo "Uploading $LOCAL_PATH to $VAST_USER@$VAST_IP:$REMOTE_PATH ..."
rsync -avz --delete -e "ssh -i $VAST_KEY -p $VAST_PORT" \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.log' \
    "$LOCAL_PATH" \
    "$VAST_USER@$VAST_IP:$REMOTE_PATH"

echo "Upload complete!"
