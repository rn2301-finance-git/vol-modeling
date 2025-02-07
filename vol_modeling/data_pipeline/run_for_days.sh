#!/bin/sh

start_date="20240120"
end_date="20250114"

current_date=$(date -d "$start_date" +%Y%m%d)

while [ "$current_date" -le "$end_date" ]; do
    day_of_week=$(date -d "$current_date" +%u)

    if [ "$day_of_week" -le 5 ]; then
        echo "Processing: $current_date"
        python merge_vol_data.py -s "$current_date" -e "$current_date" 
        python extract_top10_data.py -s "$current_date" -e "$current_date" 
    fi

    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done
