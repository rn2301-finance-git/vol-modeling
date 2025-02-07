# aliases.sh
#!/bin/bash

CONFIG_DIR="$HOME/.vol_project_config"
INSTANCES_FILE="$CONFIG_DIR/instances.json"

# Create config directory and JSON file if they don't exist
mkdir -p "$CONFIG_DIR"
if [ ! -f "$INSTANCES_FILE" ]; then
    echo '{}' > "$INSTANCES_FILE"
fi

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is required. Install with: brew install jq"
    exit 1
fi

function list_instances() {
    echo "Available cloud instances:"
    jq -r 'to_entries | .[] | "\(.key):\n  Host: \(.value.host)\n  Port: \(.value.port)\n  User: \(.value.user)\n  Auth: \(.value.auth_type) (\(.value.auth_file))\n  Provider: \(.value.provider)\n  Remote Dir: \(.value.remote_dir)\n  Description: \(.value.description)\n"' "$INSTANCES_FILE"
}

function add_instance() {
    read -p "Enter alias name: " alias
    read -p "Enter host (IP/DNS): " host
    read -p "Enter port [22]: " port
    port=${port:-22}
    read -p "Enter username: " user
    read -p "Enter auth type (key/password): " auth_type
    
    if [ "$auth_type" = "key" ]; then
        read -p "Enter key file path: " auth_file
    else
        auth_file="n/a"
    fi
    
    read -p "Enter cloud provider: " provider
    read -p "Enter remote directory path: " remote_dir
    read -p "Enter description: " description
    
    # Add new instance to JSON
    jq --arg alias "$alias" \
       --arg host "$host" \
       --arg port "$port" \
       --arg user "$user" \
       --arg auth_type "$auth_type" \
       --arg auth_file "$auth_file" \
       --arg provider "$provider" \
       --arg remote_dir "$remote_dir" \
       --arg desc "$description" \
       '. + {($alias): {
           "host": $host,
           "port": $port,
           "user": $user,
           "auth_type": $auth_type,
           "auth_file": $auth_file,
           "provider": $provider,
           "remote_dir": $remote_dir,
           "description": $desc
       }}' "$INSTANCES_FILE" > tmp.json && mv tmp.json "$INSTANCES_FILE"
    
    echo "Instance added!"
}

function remove_instance() {
    read -p "Enter alias to remove: " alias
    jq "del(.[\"$alias\"])" "$INSTANCES_FILE" > tmp.json && mv tmp.json "$INSTANCES_FILE"
    echo "Instance removed!"
}

function get_instance_details() {
    local alias=$1
    jq -r --arg alias "$alias" '.[$alias]' "$INSTANCES_FILE"
}

case "$1" in
    "list")
        list_instances
        ;;
    "add")
        add_instance
        ;;
    "remove")
        remove_instance
        ;;
    *)
        echo "Usage: $0 {list|add|remove}"
        echo "Manages cloud instance aliases for vol_project project"
        ;;
esac