#!/bin/bash

start_date=$1
end_date=$2
experiment_name=$3
prefix=$4
current_date="$start_date"

while [ "$(date -d "$current_date" +%Y%m%d)" -le "$(date -d "$end_date" +%Y%m%d)" ]; do
    # Check if the current date is a weekday (Monday to Friday)
    day_of_week=$(date -d "$current_date" +%u)
    if [ "$day_of_week" -lt 6 ]; then
        dt=$(date -d "$current_date" +%Y%m%d)
        echo "Running inference for date: $dt"
        python run_inference.py -m transformer -e "$experiment_name" -p "$prefix" -d "$dt" --no-overwrite
    fi
    # Move to the next day
    current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
done
