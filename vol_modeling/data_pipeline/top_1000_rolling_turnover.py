import boto3
import re
import os
from datetime import date, datetime
import pandas as pd
import databento as db
import warnings
import argparse
# --------------------------------------------------------------------
# Configuration variables (edit as needed)
#
S3_BUCKET = "volatility-project"
# This is where your daily OHLCV data is stored
# e.g.: "databento/OHLCV/2025/01/15/data"
S3_OHLCV_PREFIX = "databento/OHLCV"

# Where we'll upload the daily top-1000 turnover CSV files
# e.g. "data/top1000_daily_turnover/2025/01/"
S3_OUTPUT_PREFIX = "data/daily_turnover"

# --------------------------------------------------------------------


def parse_date_from_key(key: str) -> date:
    """
    Extract the date from an S3 key name like:
      'databento/OHLCV/2025/01/15/data/xnas-itch-20250115.ohlcv-1d.dbn.zst'
    Return a Python date (e.g. 2025-01-15).
    """
    # This regex looks for xnas-itch-YYYYMMDD.ohlcv-1d.dbn.zst
    pattern = r"xnas-itch-(\d{8})\.ohlcv-1d\.dbn\.zst$"
    match = re.search(pattern, key)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return None


def list_ohlcv_files_in_s3(
    s3_bucket: str, 
    s3_prefix: str
) -> pd.DataFrame:
    """
    List all S3 objects under the given prefix. 
    For each `.dbn.zst` that matches our pattern, parse out the date.

    Returns a DataFrame with columns = ["s3_key", "file_date"], sorted by date.
    """

    # Initialize Boto3 S3 client
    s3_client = boto3.client("s3")

    # We'll collect results in a list of (key, date)
    records = []
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".dbn.zst"):
                fdate = parse_date_from_key(key)
                if fdate is not None:
                    records.append((key, fdate))

    df_files = pd.DataFrame(records, columns=["s3_key", "file_date"])
    df_files.sort_values("file_date", inplace=True)
    df_files.reset_index(drop=True, inplace=True)
    return df_files


def load_ohlc_data_from_s3(s3_bucket: str, s3_key: str) -> pd.DataFrame:
    """
    Load a single day's OHLC data from S3 using the databento DBNStore,
    then compute "turnover" = volume * close.
    """
    # Download the file in-memory (or to a temporary file). We'll do in-memory here.
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    file_bytes = response["Body"].read()

    # We can feed the bytes directly into DBNStore:
    store = db.DBNStore.from_bytes(file_bytes)
    df_ohlc = store.to_df()
    df_ohlc["turnover"] = df_ohlc["volume"] * df_ohlc["close"]
    return df_ohlc


def compute_rolling_average_turnover_excluding_current_day(
    s3_bucket: str,
    s3_ohlcv_prefix: str,
    s3_output_prefix: str,
    start_date: date,
    end_date: date,
    lookback_days: int = 10,
    top_n: int = None  # Made optional, defaults to None for all stocks
):
    """
    For each trading day D in [start_date, end_date], compute the average
    turnover from the preceding `lookback_days` trading days (excluding D),
    then write out a CSV to S3 with all stocks ranked by turnover.

    Output CSV includes:
    - symbol
    - avg_turnover
    - rank (1 = highest turnover)
    - num_days (number of days with data in the lookback window)
    """

    # 1) List all daily files from S3
    df_files = list_ohlcv_files_in_s3(s3_bucket, s3_ohlcv_prefix)

    # Build a dict of date -> s3_key for quick lookup
    date_to_s3key = dict(zip(df_files["file_date"], df_files["s3_key"]))

    # Sort all trading dates
    all_trading_dates = sorted(date_to_s3key.keys())

    rolling_data = []  # list of (day_date, DataFrame) for preceding days

    # We'll iterate all trading dates in ascending order,
    # building up the rolling window. 
    for day_date in all_trading_dates:

        # Remove older-than-10-day data if the buffer is too big
        while len(rolling_data) > lookback_days:
            rolling_data.pop(0)

        # If day_date is within the date range we want to produce CSV for...
        if start_date <= day_date <= end_date:
            # Use the existing rolling_data to compute average turnover
            if len(rolling_data) == 0:
                # no preceding data => skip or produce empty
                print(f"No preceding data for {day_date}, skipping output.")
            else:
                # Merge and compute average turnover with day count
                df_rolling = pd.concat([df for (_, df) in rolling_data], ignore_index=True)
                df_avg_turnover = (
                    df_rolling
                    .groupby("symbol")
                    .agg({
                        "turnover": "mean",
                        "symbol": "count"  # Count occurrences for num_days
                    })
                    .rename(columns={
                        "turnover": "avg_turnover",
                        "symbol": "num_days"
                    })
                    .reset_index()
                )

                # Sort descending by turnover and add rank
                df_ranked = df_avg_turnover.sort_values("avg_turnover", ascending=False)
                df_ranked["rank"] = range(1, len(df_ranked) + 1)

                # If top_n is specified, filter to top N stocks
                if top_n is not None:
                    df_ranked = df_ranked.head(top_n)

                # Generate S3 key for output
                out_key = f"s3://{s3_bucket}/{s3_output_prefix}/{day_date:%Y%m%d}_turnover.csv"
                df_ranked.to_csv(out_key, index=False)
                print(f"Uploaded CSV to {out_key}")

        # Finally, load *this* day's data (if any) and add to rolling window
        # so it's used for subsequent days
        if day_date > end_date:
            # We're done if day_date is past the range we care about
            break

        # Load the day's data from S3
        df_current = load_ohlc_data_from_s3(s3_bucket, date_to_s3key[day_date])
        rolling_data.append((day_date, df_current))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute rolling average turnover for stocks')
    
    # Date arguments
    parser.add_argument('--start-date', '-s',
                      required=True,
                      help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', '-e',
                      required=True,
                      help='End date in YYYYMMDD format')
    
    # Optional arguments
    parser.add_argument('--lookback-days', '-l',
                      type=int,
                      default=10,
                      help='Number of lookback days for rolling average (default: 10)')
    parser.add_argument('--top-n', '-n',
                      type=int,
                      default=None,
                      help='Optional: limit output to top N stocks by turnover')
    
    # S3 configuration
    parser.add_argument('--s3-bucket', '-b',
                      default=S3_BUCKET,
                      help=f'S3 bucket name (default: {S3_BUCKET})')
    parser.add_argument('--s3-ohlcv-prefix', '-i',
                      default=S3_OHLCV_PREFIX,
                      help=f'S3 prefix for OHLCV data (default: {S3_OHLCV_PREFIX})')
    parser.add_argument('--s3-output-prefix', '-o',
                      default=S3_OUTPUT_PREFIX,
                      help=f'S3 prefix for output files (default: {S3_OUTPUT_PREFIX})')

    args = parser.parse_args()

    # Parse dates from strings
    try:
        start_date = datetime.strptime(args.start_date, "%Y%m%d").date()
        end_date = datetime.strptime(args.end_date, "%Y%m%d").date()
    except ValueError as e:
        parser.error(f"Invalid date format. Please use YYYYMMDD. Error: {e}")

    # Validate date range
    if end_date < start_date:
        parser.error("End date must be after start date")

    compute_rolling_average_turnover_excluding_current_day(
        s3_bucket=args.s3_bucket,
        s3_ohlcv_prefix=args.s3_ohlcv_prefix,
        s3_output_prefix=args.s3_output_prefix,
        start_date=start_date,
        end_date=end_date,
        lookback_days=args.lookback_days,
        top_n=args.top_n
    )
