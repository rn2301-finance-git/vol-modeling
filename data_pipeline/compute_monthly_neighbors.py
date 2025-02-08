import argparse
import sys
import boto3
import s3fs
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from io import StringIO
import calendar
import os

BUCKET_NAME = os.environ.get('BUCKET_NAME')
if not BUCKET_NAME:
    raise ValueError("Environment variable BUCKET_NAME must be set")

def list_months(start_ym, end_ym):
    """
    Given start_ym='YYYYMM' and end_ym='YYYYMM',
    return a list of 'YYYYMM' in ascending order.
    Example: start_ym='202401', end_ym='202403' -> ['202401', '202402', '202403'].
    """
    start_dt = datetime.strptime(start_ym, "%Y%m")
    end_dt = datetime.strptime(end_ym, "%Y%m")
    if end_dt < start_dt:
        raise ValueError("end_ym cannot be before start_ym")

    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime("%Y%m"))
        # move to next month
        year = current.year
        month = current.month
        if month == 12:
            next_month_dt = datetime(year=year+1, month=1, day=1)
        else:
            next_month_dt = datetime(year=year, month=month+1, day=1)
        current = next_month_dt
    return months


def get_prev_month_ym(ym_str):
    """
    Return the previous month of ym_str='YYYYMM'.
    If ym_str='202401', returns '202312'.
    """
    dt = datetime.strptime(ym_str, "%Y%m")
    year = dt.year
    month = dt.month
    if month == 1:
        prev_m_dt = datetime(year-1, 12, 1)
    else:
        prev_m_dt = datetime(year, month-1, 1)
    return prev_m_dt.strftime("%Y%m")


def list_days_in_month(ym_str):
    """
    Return a list of 'YYYYMMDD' for all days in the given month ym_str='YYYY-MM'.
    """
    dt = datetime.strptime(ym_str, "%Y%m")
    year = dt.year
    month = dt.month
    num_days = calendar.monthrange(year, month)[1]  # days in that month
    out = []
    for day in range(1, num_days+1):
        out.append(datetime(year, month, day).strftime("%Y%m%d"))
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute nearest neighbors for a month from prior month's log_ret_1m.")
    parser.add_argument("--month", required=True, help="Month in YYYYMM format")
    parser.add_argument("--top_k", type=int, default=5, help="Number of neighbors to compute")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="S3 bucket name")
    parser.add_argument("--check", action="store_true", help="Print debug info")
    args = parser.parse_args()

    month_str = args.month
    top_k = args.top_k
    bucket = args.bucket
    print_debug = args.check

    # No need for list_months since we're only processing one month
    print("Processing month:", month_str)

    # 2) We'll gather the daily parquet keys from S3 for the previous month
    s3_fs = s3fs.S3FileSystem()
    prefix_itch = "data/features/itch"

    # We'll map day->key
    # Keys look like: data/features/itch/20240116.itch_features.parquet
    # or data/features/itch/{YYYYMMDD}.itch_features.parquet
    # We can list that entire prefix and parse the filenames.
    print("Listing S3 keys under", prefix_itch, "...")
    all_keys = s3_fs.ls(f"{bucket}/{prefix_itch}")
    # all_keys is a list of dicts or strings, depending on s3fs version. Typically a list of "bucket/prefix/file"
    # We'll parse out the date from the filename. A typical key might be "volatility-project/data/features/itch/20240116.itch_features.parquet"

    day_to_key = {}
    for k in all_keys:
        # in newer s3fs versions, k might be just a string "<bucket>/<prefix>/filename"
        # or an dict with 'Key' key. Let's unify by ensuring k is a string
        if isinstance(k, dict):
            k = k["Key"]  # S3FileInfo dict?
        # we want to extract something like 20240116 from the last part
        match = re.search(r"(\d{8})\.itch_features\.parquet$", k)
        if match:
            dt_str = match.group(1)
            day_to_key[dt_str] = k  # store full path

    print(f"Found {len(day_to_key)} daily parquet files in S3 (parsed).")

    # 3) For each month in [start_ym..end_ym], we want to read the previous month data
    #    If previous month has no data, we'll skip or produce an empty neighbor file.
    s3_client = boto3.client("s3")

    prev_month_str = get_prev_month_ym(month_str)
    # E.g. if month_str='202403', prev_month_str='202402'

    # Let's gather daily data from prev_month_str
    prev_month_days = list_days_in_month(prev_month_str)
    # e.g. ["20240201", "20240202", ..., "20240228"]

    # If there's no overlap or no data at all, just skip or produce empty
    # We'll load all daily parquets for these days into one big DataFrame, then pivot.
    df_list = []
    for d_str in prev_month_days:
        if d_str in day_to_key:
            parquet_key = day_to_key[d_str]  # full path from s3fs.ls()
            s3_uri = f"s3://{parquet_key}"

            try:
                df_day = pd.read_parquet(s3_uri, filesystem=s3_fs)
                df_list.append(df_day[["date", "minute", "symbol", "log_ret_1m"]])
            except Exception as e:
                print(f"Warning: failed to read {s3_uri}: {e}")
        else:
            pass


    if not df_list:
        print(f"No data found for previous month {prev_month_str}. Skipping neighbors for {month_str}...")
        return

    df_prev_month = pd.concat(df_list, axis=0, ignore_index=True)
    if df_prev_month.empty:
        print(f"Empty DataFrame for {prev_month_str}, skipping neighbors for {month_str}")
        return

    # 4) Pivot to get time x symbol matrix of log_ret_1m
    # We'll combine date+minute into a single string or a datetime if you prefer.
    df_prev_month["datetime_str"] = df_prev_month["date"].astype(str) + "_" + df_prev_month["minute"]
    pivot_df = df_prev_month.pivot_table(
        index="datetime_str",
        columns="symbol",
        values="log_ret_1m"
    )

    # 5) Compute correlation among symbols (across time index)
    corr_matrix = pivot_df.corr(method="pearson")  # symbol x symbol

    # 6) For each symbol, find top_k correlated symbols
    symbols = corr_matrix.index
    neighbors_dict = {}
    for sym in symbols:
        # exclude self
        sorted_corr = corr_matrix[sym].drop(sym, errors="ignore").sort_values(ascending=False)
        top_syms = sorted_corr.head(top_k).index.tolist()
        neighbors_dict[sym] = top_syms

    # 7) Save to S3 as CSV:  [symbol, n1, n2, ..., nK]
    output_rows = []
    columns = ["symbol"] + [f"n{i}" for i in range(1, top_k+1)]
    for sym, nbrs in neighbors_dict.items():
        row = [sym] + nbrs
        output_rows.append(row)
    nn_df = pd.DataFrame(output_rows, columns=columns)

    # We'll store it in e.g. data/features/nn/202403_neighbors.csv
    # meaning "neighbors to use for the month 202403"
    out_key = f"data/features/nn/{month_str}_neighbors.csv"
    csv_buf = StringIO()
    nn_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=out_key,
            Body=csv_bytes
        )
        print(f"Saved neighbors for {month_str} => s3://{bucket}/{out_key} (Top {top_k})")
        if print_debug:
            print(nn_df.head())
    except Exception as e:
        print(f"Error uploading neighbors for {month_str}: {e}")

    print("\nDone! Monthly neighbors computed.")


if __name__ == "__main__":
    main()
