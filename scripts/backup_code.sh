#!/bin/bash

# Get timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p ~/Backups/BAM_backups

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
    ~/Code/BAM/ ~/Backups/BAM_backups/BAM_${timestamp}/

# Print backup size
echo "Created backup: BAM_${timestamp}"
echo "Backup size:"
du -sh ~/Backups/BAM_backups/BAM_${timestamp}

# Keep only last 5 backups (optional)
#cd ~/Backups/BAM_backups && ls -t | tail -n +6 | xargs rm -rf 2>/dev/null

# List remaining backups
echo -e "\nExisting backups:"
ls -lh ~/Backups/BAM_backups