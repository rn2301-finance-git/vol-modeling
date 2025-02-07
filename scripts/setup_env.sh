#!/bin/bash

# Exit on any error
set -e

FORCE=0
CUDA=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force) FORCE=1 ;;
        --cuda) CUDA=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Setting up Python environment..."

# Check if .venv exists
if [ -d ".venv" ]; then
    if [ $FORCE -eq 1 ]; then
        echo "Recreating virtual environment..."
        rm -rf .venv
    else
        echo "Virtual environment exists. Use --force to recreate."
        echo "Current environment preserved."
        exit 0
    fi
fi

# Check Python version and availability
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD=python3.9
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD=python3.8
else
    PYTHON_CMD=python3
fi

echo "Using Python command: $PYTHON_CMD"

# Create virtual environment
$PYTHON_CMD -m venv .venv
source ./.venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Check CUDA availability if requested
if [ $CUDA -eq 1 ]; then
    echo "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA is available. Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Warning: CUDA requested but nvidia-smi not found. Installing CPU-only PyTorch."
        pip install torch torchvision torchaudio
    fi
fi

# Install requirements
if [ -f "requirements.txt" ]; then
    # Remove torch-related packages from requirements if we installed CUDA version
    if [ $CUDA -eq 1 ]; then
        echo "Filtering torch-related packages from requirements.txt..."
        grep -v "torch\|torchaudio\|torchvision" requirements.txt > requirements_filtered.txt
        pip install -r requirements_filtered.txt
        rm requirements_filtered.txt
    else
        pip install -r requirements.txt
    fi
else
    echo "No requirements.txt found!"
    exit 1
fi

echo "Environment setup complete!"
echo "Activate with: source .venv/bin/activate"
