#!/bin/bash

# Get timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p ~/Backups/vol_project_backups

# Create backup using rsync with exclusions
rsync -av --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='data' \
    --exclude='models/checkpoints' \
    --exclude='*.pth' \
    --exclude='*.h5' \
    --exclude='logs' \
    --exclude='*.log' \
    --exclude='.idea' \
    --exclude='.vscode' \
    ~/Code/vol_modeling/ ~/Backups/vol_project_backups/vol_project_${timestamp}/

# Print backup size
echo "Created backup: vol_project_${timestamp}"
echo "Backup size:"
du -sh ~/Backups/vol_project_backups/vol_project_${timestamp}

# Keep only last 5 backups (optional)
#cd ~/Backups/vol_project_backups && ls -t | tail -n +6 | xargs rm -rf 2>/dev/null

# List remaining backups
echo -e "\nExisting backups:"
ls -lh ~/Backups/vol_project_backups