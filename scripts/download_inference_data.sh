#!/bin/bash

# Check if BUCKET_NAME environment variable is set
if [ -z "$BUCKET_NAME" ]; then
    echo "Error: BUCKET_NAME environment variable must be set"
    exit 1
fi

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <model_type> <experiment_name> <run_prefix>"
    exit 1
fi

model_type="$1"
experiment_name="$2"
run_prefix="$3"
local_dir="/workspace/inference/"$model_type/$run_prefix

mkdir -p "${local_dir}"

# Print S3 path
echo "Downloading from: s3://${BUCKET_NAME}/experiments/${model_type}/${experiment_name}/"

# Get the list of matching files from S3
aws s3 ls "s3://${BUCKET_NAME}/experiments/${model_type}/${experiment_name}/" --recursive | grep "${run_prefix}.*/inference/.*\.5min\.csv" | while read -r line; do
    # Extract the file path from the listing
    file_path=$(echo "$line" | awk '{print $NF}')
    local_file="${local_dir}/$(basename "$file_path")"

    # Check if the file already exists locally
    if [ ! -f "$local_file" ]; then
        echo "Downloading: $file_path"
        aws s3 cp "s3://${BUCKET_NAME}/$file_path" "$local_file"
    else
        echo "Skipping existing file: $local_file"
    fi
done
