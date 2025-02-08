#!/bin/bash

CONFIG_DIR="$HOME/.vol_project_config"
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

# Build and execute SSH command based on auth type
if [ "$AUTH_TYPE" = "key" ]; then
    ssh -i "$AUTH_FILE" -p "$PORT" "$USER@$HOST"
else
    ssh -p "$PORT" "$USER@$HOST"
fi