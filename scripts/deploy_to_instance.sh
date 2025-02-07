#!/bin/bash

# Exit on any error
set -e

# Directory setup
CONFIG_DIR="$HOME/.bam_config"
INSTANCES_FILE="$CONFIG_DIR/instances.json"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

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
BASE_REMOTE_DIR=$(echo "$INSTANCE_INFO" | jq -r '.remote_dir')
REMOTE_DIR="${BASE_REMOTE_DIR}/BAM"

echo "Deploying to instance at $HOST (directory: $REMOTE_DIR)"

# Build SSH and RSYNC commands based on auth type
if [ "$AUTH_TYPE" = "key" ]; then
    SSH_CMD="ssh -i \"$AUTH_FILE\" -p $PORT"
    RSYNC_AUTH="-e \"ssh -i $AUTH_FILE -p $PORT\""
else
    SSH_CMD="ssh -p $PORT"
    RSYNC_AUTH="-e \"ssh -p $PORT\""
fi

# First run backup script
echo "Creating backup..."
"$SCRIPTS_DIR/backup_code.sh"

# Get latest backup
LATEST_BACKUP=$(ls -t "$HOME/BAM_backups" | head -n 1)
echo "Using backup: $LATEST_BACKUP"

# Ensure the remote directory exists
eval "$SSH_CMD \"$USER@$HOST\" \"mkdir -p $REMOTE_DIR\""

# Copy the latest backup
echo "Copying files to instance..."
eval "rsync -avz --delete $RSYNC_AUTH \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='requirements.txt' \
    \"$HOME/BAM_backups/$LATEST_BACKUP/\" \
    \"$USER@$HOST:$REMOTE_DIR/\""

# Copy the setup script
echo "Copying setup script..."
eval "rsync -avz $RSYNC_AUTH \"$SCRIPTS_DIR/setup_env.sh\" \"$USER@$HOST:$REMOTE_DIR/\""

# Make setup script executable
eval "$SSH_CMD \"$USER@$HOST\" \"chmod +x $REMOTE_DIR/setup_env.sh\""

# Ask about environment setup
echo
read -p "Would you like to set up the Python environment now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if CUDA should be installed
    read -p "Install with CUDA support? (y/n) " -n 1 -r
    echo
    CUDA_FLAG=""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CUDA_FLAG="--cuda"
    fi
    
    # Run setup script
    eval "$SSH_CMD \"$USER@$HOST\" \"cd $REMOTE_DIR && ./setup_env.sh $CUDA_FLAG\""
fi

echo
echo "Deployment complete!"
echo "To access your instance:"
echo "1. SSH into your instance: $SSH_CMD $USER@$HOST"
echo "2. cd $REMOTE_DIR"
echo
read -p "Would you like to SSH into the instance now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    eval "$SSH_CMD \"$USER@$HOST\""
fi 