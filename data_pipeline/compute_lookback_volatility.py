import argparse
import sys
import boto3
import s3fs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from tempfile import NamedTemporaryFile
from feature_engineering import get_dates_from_s3

BUCKET_NAME = os.environ.get('BUCKET_NAME')

def date_range_list(start_str, end_str):
    """
    Return list of date strings [start_str, ..., end_str] inclusive,
    in ascending order, naive daily increments.
    """
    fmt = "%Y%m%d"
    start_dt = datetime.strptime(start_str, fmt)
    end_dt = datetime.strptime(end_str, fmt)
    out = []
    curr = start_dt
    while curr <= end_dt:
        out.append(curr.strftime(fmt))
        curr += timedelta(days=1)
    return out

def list_prior_dates(dt_str, n=10, bucket=BUCKET_NAME, prefix="data/features/itch"):
    """
    Return up to N available trading days before dt_str (not including dt_str), in ascending order.
    Uses actual S3 data availability rather than calendar days.
    Example: If dt_str=20240116 and we have [20240112, 20240115, 20240116] in S3,
    with n=2 this returns [20240112, 20240115]
    """
    # Get all available dates from S3
    all_dates = get_dates_from_s3(bucket, prefix)
    all_dates = sorted(all_dates)  # Ensure ascending order
    
    # Find the position of our target date
    try:
        current_idx = all_dates.index(dt_str)
    except ValueError:
        return []  # Target date not found
    
    # Get up to n prior dates, excluding current date
    start_idx = max(0, current_idx - n)
    prior_dates = all_dates[start_idx:current_idx]  # Excludes current date
    
    return prior_dates

def compute_daily_avg_10m_vol(parquet_path, s3_fs):
    """
    Reads a day's feature Parquet from S3, computes each symbol's
    average 10-min volatility (mean of target_vol_10m), returns a Series:
       symbol -> daily_avg_vol
    If the file doesn't exist or is empty, returns None.
    """
    try:
        df_day = pd.read_parquet(parquet_path, filesystem=s3_fs)
        if len(df_day) == 0:
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return None

    # If your realized forward 10-minute vol is in "target_vol_10m"
    daily_vol = df_day.groupby("symbol")["target_vol_10m"].mean()
    return daily_vol

def main():
    parser = argparse.ArgumentParser(
        description="Compute average intraday 10-min vol for each date in a range, without re-reading prior days multiple times."
    )
    parser.add_argument("-s", "--start_date", required=True,
                        help="Start date in YYYYMMDD format")
    parser.add_argument("-e", "--end_date", required=True,
                        help="End date in YYYYMMDD format")
    parser.add_argument("-c", "--check", action="store_true",
                        help="Print some debug info (sanity checks)")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip)")

    args = parser.parse_args()
    start_str = args.start_date
    end_str = args.end_date
    print_check = args.check

    # Validate date formats
    try:
        start_dt = datetime.strptime(start_str, "%Y%m%d")
        end_dt = datetime.strptime(end_str, "%Y%m%d")
    except ValueError:
        print("Error: Invalid date format. Please use YYYYMMDD (e.g. 20240116)")
        sys.exit(1)

    if end_dt < start_dt:
        print("Error: end_date cannot be before start_date")
        sys.exit(1)

    print(f"\nComputing average intraday vol from {start_str} to {end_str} (inclusive)...")

    s3_fs = s3fs.S3FileSystem()
    bucket = BUCKET_NAME
    prefix_itch = "data/features/itch"
    prefix_out = "data/features/intraday_volatility"

    # Get actual trading days that exist in S3
    all_dates = get_dates_from_s3(bucket, prefix_itch)
    target_dates = sorted([d for d in all_dates if start_str <= d <= end_str])
    
    if not target_dates:
        print(f"No trading days found between {start_str} and {end_str}")
        sys.exit(1)
    
    print(f"Found {len(target_dates)} trading days to process")

    # Instead of loading all dates at once, we'll maintain a rolling window
    day_to_vol = dict()  # Rolling window cache: "YYYYMMDD" -> pd.Series(symbol->avg 10min vol)
    
    def load_lookback_data(target_date, lookback_days=10):
        """
        Load/maintain rolling window of lookback data for target_date.
        Returns list of dates actually loaded (for progress reporting).
        """
        # Get the dates we need for this target date
        needed_dates = list_prior_dates(target_date, lookback_days)
        
        # Remove dates from cache that we no longer need
        current_dates = list(day_to_vol.keys())
        for old_date in current_dates:
            if old_date not in needed_dates:
                del day_to_vol[old_date]
        
        # Load any missing dates we need
        loaded_dates = []
        for date_str in needed_dates:
            if date_str not in day_to_vol:
                path = f"s3://{bucket}/{prefix_itch}/{date_str}.itch_features.parquet"
                day_vol = compute_daily_avg_10m_vol(path, s3_fs)
                if day_vol is not None and len(day_vol) > 0:
                    day_to_vol[date_str] = day_vol
                    loaded_dates.append(date_str)
        
        return loaded_dates

    # We'll go day by day in ascending order
    target_dates = date_range_list(start_str, end_str)

    def get_multi_day_avg_vol(symbol, needed_dates):
        # If ANY needed date is missing from day_to_vol, return NaN
        # If symbol is missing for any day, return NaN
        sum_ = 0.0
        count_ = 0
        for d_ in needed_dates:
            if d_ not in day_to_vol:
                return np.nan
            s_ = day_to_vol[d_]
            if symbol not in s_.index:
                return np.nan
            sum_ += s_[symbol]
            count_ += 1
        return sum_ / count_ if count_ > 0 else np.nan

    s3_client = boto3.client("s3")
    for dt_str in target_dates:
        # Check if output already exists
        out_filename = f"{dt_str}.average_intraday_volatility.parquet"
        output_key = f"{prefix_out}/{out_filename}"
        
        try:
            s3_client.head_object(Bucket=bucket, Key=output_key)
            if not args.overwrite:
                print(f"Output exists for {dt_str}, skipping (use -o to overwrite)")
                continue
            print(f"Output exists for {dt_str}, will overwrite")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"\nProcessing day {dt_str}...")
            else:
                print(f"Error checking {output_key}: {e}")
                continue
        
        # Load/update rolling window of lookback data
        loaded = load_lookback_data(dt_str)
        if loaded:
            print(f"Loaded {len(loaded)} new lookback dates: {loaded}")

        # Step A: read dt_str's feature file (so we get the symbol set)
        path_key = f"{prefix_itch}/{dt_str}.itch_features.parquet"
        s3_uri = f"s3://{bucket}/{path_key}"
        try:
            with NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
                tmp_path = tmp_file.name
                s3_client.download_file(bucket, path_key, tmp_path)
                df_day = pd.read_parquet(tmp_path)
            os.unlink(tmp_path)
        except FileNotFoundError:
            print(f"Warning: No parquet found for {dt_str}, skipping.")
            continue
        except Exception as e:
            print(f"Error reading {s3_uri}: {e}")
            continue

        if len(df_day) == 0:
            print(f"Empty DataFrame for {dt_str}, skipping.")
            continue

        # Step B: build output for each symbol in df_day
        # We want columns [symbol, date, avg_vol_1d, avg_vol_2d, avg_vol_5d, avg_vol_10d].
        symbols_today = df_day["symbol"].unique()
        out_data = {
            "symbol": [],
            "avg_vol_1d": [],
            "avg_vol_2d": [],
            "avg_vol_5d": [],
            "avg_vol_10d": [],
        }

        # Pre-build the offset date lists
        # dt-1..dt-1
        day_1 = list_prior_dates(dt_str, 1)
        # dt-1..dt-2
        day_2 = list_prior_dates(dt_str, 2)
        # dt-1..dt-5
        day_5 = list_prior_dates(dt_str, 5)
        # dt-1..dt-10
        day_10 = list_prior_dates(dt_str, 10)

        for sym in symbols_today:
            out_data["symbol"].append(sym)

            out_data["avg_vol_1d"].append(get_multi_day_avg_vol(sym, day_1))
            out_data["avg_vol_2d"].append(get_multi_day_avg_vol(sym, day_2))
            out_data["avg_vol_5d"].append(get_multi_day_avg_vol(sym, day_5))
            out_data["avg_vol_10d"].append(get_multi_day_avg_vol(sym, day_10))

        df_out = pd.DataFrame(out_data)

        if print_check:
            print(f"Output sample for {dt_str}:\n{df_out.head(5)}")

        # Step C: save to S3
        out_filename = f"{dt_str}.average_intraday_volatility.parquet"
        output_key = f"{prefix_out}/{out_filename}"
        s3_uri_out = f"s3://{bucket}/{output_key}"
        try:
            with NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
                temp_path = tmp_file.name
                df_out.to_parquet(temp_path, index=False)
                s3_client.upload_file(temp_path, bucket, output_key)
            os.unlink(temp_path)
            print(f"Saved {len(df_out)} rows to {s3_uri_out}")
        except Exception as e:
            print(f"Error saving {s3_uri_out}: {e}")

    print("\nAll done!")


if __name__ == "__main__":
    if not BUCKET_NAME:
        print("Error: BUCKET_NAME environment variable not set")
        sys.exit(1)
    main()
