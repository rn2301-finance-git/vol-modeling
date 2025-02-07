#!/usr/bin/env python3
"""
Script to update target field names in pre‐computed parquet files on S3.

For files created under:
  - data/features/attention_df/all/
  - data/features/attention_df/top_n/

this script will scan all files for dates within the provided range and,
if found, rename the columns from (for example)
    Y_vol_10min           -> Y_log_vol_10min
    Y_ret_10min           -> Y_log_ret_10min
    Y_ret_30min           -> Y_log_ret_30min
    Y_ret_60min           -> Y_log_ret_60min
    Y_ret_10min_partial   -> Y_log_ret_10min_partial
    Y_ret_30min_partial   -> Y_log_ret_30min_partial
    Y_ret_60min_partial   -> Y_log_ret_60min_partial
    Y_vol_10min_lag_1m     -> Y_log_vol_10min_lag_1m
    Y_ret_10min_lag_1m     -> Y_log_ret_10min_lag_1m
    Y_ret_30min_lag_1m     -> Y_log_ret_30min_lag_1m
    Y_ret_60min_lag_1m     -> Y_log_ret_60min_lag_1m

If no update flag is provided, the script updates files in both directories.
"""

import argparse
from datetime import datetime
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def update_parquet_file(file_path, rename_dict, fs):
    """Reads a parquet file from S3, renames columns per rename_dict, prints a sanity check head, and writes back."""
    print(f"\nReading file: {file_path}")
    try:
        with fs.open(file_path, 'rb') as f:
            table = pq.read_table(f)
            df = table.to_pandas()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    print("=== Original file head ===")
    print(df.head())

    # Determine which columns in the file need to be renamed.
    columns_to_rename = {old: new for old, new in rename_dict.items() if old in df.columns}
    if not columns_to_rename:
        print("No target columns found to rename in this file.")
    else:
        print(f"Renaming columns: {columns_to_rename}")
        df.rename(columns=columns_to_rename, inplace=True)

    print("=== Updated file head ===")
    print(df.head())

    try:
        with fs.open(file_path, 'wb') as f:
            new_table = pa.Table.from_pandas(df)
            pq.write_table(new_table, f, compression='snappy')
        print(f"Updated file: {file_path}")
    except Exception as e:
        print(f"Error writing {file_path}: {e}")

def process_files(prefix, pattern, start_dt, end_dt, rename_dict, bucket, fs):
    """
    Lists all parquet files matching the full S3 pattern,
    filters by date (assumes the file name begins with YYYYMMDD),
    and applies the update_parquet_file() function.
    """
    search_pattern = f"s3://{bucket}/{prefix}{pattern}"
    files = fs.glob(search_pattern)
    print(f"\nFound {len(files)} files in {search_pattern}")
    
    for file_path in files:
        filename = file_path.split('/')[-1]
        # Assume file names are either "YYYYMMDD.parquet" or "YYYYMMDD.top*.parquet"
        parts = filename.split('.')
        if len(parts) < 2:
            print(f"Skipping file {filename} (unexpected format).")
            continue
        date_str = parts[0]
        try:
            file_dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"Skipping file {filename} (date not parseable).")
            continue
        if start_dt <= file_dt <= end_dt:
            update_parquet_file(file_path, rename_dict, fs)
        else:
            print(f"Skipping {filename} (date {date_str}) outside of range.")

def main():
    parser = argparse.ArgumentParser(
        description="Rename Y return/volatility fields in parquet files on S3 for a given date range."
    )
    parser.add_argument('-s', '--start-date', required=True,
                        help='Start date in YYYYMMDD format (e.g., 20240116)')
    parser.add_argument('-e', '--end-date', required=True,
                        help='End date in YYYYMMDD format (e.g., 20240131)')
    parser.add_argument('--bucket', default='bam-volatility-project',
                        help='S3 bucket name (default: bam-volatility-project)')
    parser.add_argument('--update-all', action='store_true',
                        help='Update files in data/features/attention_df/all/')
    parser.add_argument('--update-topn', action='store_true',
                        help='Update files in data/features/attention_df/top_n/')
    
    args = parser.parse_args()
    
    try:
        start_dt = datetime.strptime(args.start_date, "%Y%m%d")
        end_dt = datetime.strptime(args.end_date, "%Y%m%d")
        if end_dt < start_dt:
            raise ValueError("End date must be >= start date.")
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return

    # If neither flag is provided, update both directories.
    update_all = args.update_all or (not args.update_all and not args.update_topn)
    update_topn = args.update_topn or (not args.update_all and not args.update_topn)

    bucket = args.bucket
    fs = s3fs.S3FileSystem(anon=False)

    # Define the mapping of old column names to new names.
    rename_dict = {
        "Y_vol_10min":         "Y_log_vol_10min",
        "Y_ret_10min":         "Y_log_ret_10min",
        "Y_ret_30min":         "Y_log_ret_30min",
        "Y_ret_60min":         "Y_log_ret_60min",
        "Y_ret_10min_partial": "Y_log_ret_10min_partial",
        "Y_ret_30min_partial": "Y_log_ret_30min_partial",
        "Y_ret_60min_partial": "Y_log_ret_60min_partial",
        "Y_vol_10min_lag_1m":    "Y_log_vol_10min_lag_1m",
        "Y_ret_10min_lag_1m":    "Y_log_ret_10min_lag_1m",
        "Y_ret_30min_lag_1m":    "Y_log_ret_30min_lag_1m",
        "Y_ret_60min_lag_1m":    "Y_log_ret_60min_lag_1m"
    }

    # Process files in the "all" directory if requested.
    if update_all:
        print("\n=== Updating files in data/features/attention_df/all/ ===")
        prefix_all = "data/features/attention_df/all/"
        # Files are assumed to have names like "YYYYMMDD.parquet"
        process_files(prefix_all, "*.parquet", start_dt, end_dt, rename_dict, bucket, fs)

    # Process files in the "top_n" directory if requested.
    if update_topn:
        print("\n=== Updating files in data/features/attention_df/top_n/ ===")
        prefix_topn = "data/features/attention_df/top_n/"
        # Files are assumed to have names like "YYYYMMDD.top*.parquet"
        process_files(prefix_topn, "*.parquet", start_dt, end_dt, rename_dict, bucket, fs)

if __name__ == '__main__':
    main()
