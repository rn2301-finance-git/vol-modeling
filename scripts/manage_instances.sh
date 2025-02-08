# manage_instances.sh
#!/bin/bash

CONFIG_FILE="$HOME/.vol_project_config/ec2_aliases.json"

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is required. Install with: brew install jq"
    exit 1
fi

function list_instances() {
    echo "Available EC2 instances:"
    jq -r 'to_entries | .[] | "\(.key):\n  IP: \(.value.ip)\n  Key: \(.value.pem)\n  Description: \(.value.description)\n"' "$CONFIG_FILE"
}

function add_instance() {
    read -p "Enter alias name: " alias
    read -p "Enter EC2 IP/DNS: " ip
    read -p "Enter PEM file path: " pem
    read -p "Enter description: " description
    
    # Add new instance to JSON
    jq --arg alias "$alias" \
       --arg ip "$ip" \
       --arg pem "$pem" \
       --arg desc "$description" \
       '. + {($alias): {"ip": $ip, "pem": $pem, "description": $desc}}' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
    
    echo "Instance added!"
}

function remove_instance() {
    read -p "Enter alias to remove: " alias
    jq "del(.[\"$alias\"])" "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
    echo "Instance removed!"
}

function get_instance_ip() {
    jq -r --arg alias "$1" '.[$alias].ip' "$CONFIG_FILE"
}

function get_instance_pem() {
    jq -r --arg alias "$1" '.[$alias].pem' "$CONFIG_FILE"
}

# Main menu
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
        echo "Manages EC2 instance aliases for vol_project project"
        ;;
esac