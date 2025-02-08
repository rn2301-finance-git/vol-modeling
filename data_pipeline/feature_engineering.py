#!/usr/bin/env python3
import os
import warnings
import boto3
import botocore
import databento as db
import pytz
import pandas as pd
import numpy as np
import re
from datetime import datetime
from io import BytesIO, StringIO
import argparse
import sys

pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ipykernel.ipkernel")

# Define the UTC and EST timezones
utc = pytz.UTC
est = pytz.timezone('US/Eastern')

# --------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------

BUCKET_NAME = os.environ.get('BUCKET_NAME')
if not BUCKET_NAME:
    raise ValueError("Environment variable BUCKET_NAME must be set")

def get_dates_from_s3(bucket_name, prefix):
    """
    Scan the specified S3 bucket+prefix for turnover files of the form YYYYMMDD.turnover.csv,
    and return a sorted list of YYYYMMDD date strings.
    """
    print(f"Gathering dates from s3://{bucket_name}/{prefix} ...")
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        print("No files found in the specified path.")
        return []
    
    date_pattern = re.compile(r'\d{8}')
    dates = set()
    
    for obj in response['Contents']:
        key = obj['Key']
        match = date_pattern.search(key)
        if match:
            dates.add(match.group())
    
    all_dates = sorted(dates)
    print(f"Found {len(all_dates)} date(s) in daily_turnover.")
    return all_dates

def parquet_exists_in_s3(bucket_name, date_str):
    """
    Check if we already have a parquet file at:
        s3://bucket_name/data/features/itch/{date_str}.itch_features.parquet
    Return True/False accordingly.
    """
    s3_key = f"data/features/itch/{date_str}.itch_features.parquet"
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return True  # file exists
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False  # file does not exist
        else:
            raise  # some other error

# --------------------------------------------------------
# ORIGINAL PROCESSING FUNCTIONS
# --------------------------------------------------------

def load_and_merge_data(dt: str) -> pd.DataFrame:
    """
    Load BBO data from S3 (Databento ITCH) and merge with daily turnover data.
    We keep the top 1500 by rank, and mark those within top 1000 (with at least 5 days).
    """
    print(f"Loading data from S3 (Databento ITCH) for {dt}...")
    s3 = boto3.client("s3")

    # 1) Load the ITCH data from S3
    dbn_key = f"databento/ITCH/data/xnas-itch-{dt}.bbo-1m.dbn.zst"
    try:
        dbn_obj = s3.get_object(Bucket=BUCKET_NAME, Key=dbn_key)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"ITCH data file not found in S3: s3://{BUCKET_NAME}/{dbn_key}\n"
            f"Please ensure the Databento ITCH file exists for date {dt}"
        )

    dbn_data = BytesIO(dbn_obj["Body"].read())  # in-memory bytes

    # 2) Load the daily turnover file from S3
    turnover_key = f"data/daily_turnover/{dt}.turnover.csv"
    try:
        turnover_obj = s3.get_object(Bucket=BUCKET_NAME, Key=turnover_key)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Turnover data file not found in S3: s3://{BUCKET_NAME}/{turnover_key}\n"
            f"Please ensure the daily turnover file exists for date {dt}"
        )

    turnover_csv = turnover_obj["Body"].read().decode("utf-8")
    turnover_df = pd.read_csv(StringIO(turnover_csv))

    # 3) Select top 1500 by rank and create in_top1000_universe
    turnover_df = turnover_df[turnover_df["rank"] <= 1500].copy()
    turnover_df["in_top1000_universe"] = (
        (turnover_df["rank"] <= 1000) & (turnover_df["num_days"] >= 5)
    ).astype(int)

    # 4) Merge
    df = db.DBNStore.from_bytes(dbn_data.getvalue()).to_df()  # raw DataFrame
    df = pd.merge(
        df,
        turnover_df[["symbol", "avg_turnover", "rank", "num_days", "in_top1000_universe"]],
        on="symbol",
        how="inner"  # keep only top-1500 symbols
    )
    print(f"Data loading complete for {dt}")

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data with timezone conversion."""
    print("Starting data preprocessing...")
    
    if 'ts_event' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Missing required column: ts_event. Please check the data format.")
    
    if not pd.api.types.is_datetime64_any_dtype(df['ts_event']):
        print("Converting ts_event to datetime...")
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
    df['ts_event'] = df['ts_event'].dt.tz_convert(est)
    
    df['date'] = df['ts_event'].dt.date
    df = df.sort_values(['symbol', 'ts_event'])
    print("Preprocessing complete")
    return df

def fill_minute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill data to minute frequency for each symbol."""
    print("Starting minute data filling...")
    cols_to_fill = [
        "side", "price", "size",
        "bid_px_00", "ask_px_00", "bid_sz_00",
        "ask_sz_00", "bid_ct_00", "ask_ct_00",
        "avg_turnover",
        "rank", "num_days", "in_top1000_universe" 
    ]

    df_list = []
    for (sym, d), grp in df.groupby(['symbol', 'date']):
        grp = grp.set_index('ts_event')
        grp = grp.resample('1min', closed='right', label='right').last()
        
        day_start = pd.Timestamp(d).tz_localize(est) + pd.Timedelta(hours=9, minutes=30)
        day_end = pd.Timestamp(d).tz_localize(est) + pd.Timedelta(hours=16)
        
        rng = pd.date_range(start=day_start, end=day_end, freq='min', tz=est)
        grp = grp.reindex(rng)
        
        grp[cols_to_fill] = grp[cols_to_fill].ffill()
        grp[cols_to_fill] = grp[cols_to_fill].infer_objects(copy=False)

        grp['symbol'] = sym
        grp['date'] = d
        
        df_list.append(grp)

    df_filled = pd.concat(df_list, axis=0)
    df_filled = df_filled.reset_index().rename(columns={'index': 'ts_event'})
    df_filled['log_turnover'] = np.log(df_filled['avg_turnover'])
    
    print("Minute data filling complete")
    return df_filled

def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to trading hours and format minute column."""
    print("Filtering trading hours...")
    df['minute'] = df['ts_event'].dt.strftime('%H:%M')
    df = df[
        (df['minute'] >= '09:30') &
        (df['minute'] <= '16:00')
    ]
    print("Trading hours filtering complete")
    return df

def calculate_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mid price, spread and imbalance."""
    print("Calculating market metrics...")
    df["mid"] = 0.5 * (df["bid_px_00"] + df["ask_px_00"])
    df["spread"] = (df["ask_px_00"] - df["bid_px_00"]) / df["mid"]
    df["imbalance"] = (
        df["bid_sz_00"] - df["ask_sz_00"]
    ) / (df["bid_sz_00"] + df["ask_sz_00"])
    print("Market metrics calculation complete")
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns at various horizons."""
    print("Starting returns calculation...")
    def calc_log_return(series):
        return np.log(series / series.shift(1))

    df = df.reset_index(drop=True)

    # 1-minute returns
    df["log_ret_1m"] = (
        df.groupby("symbol")["mid"]
        .apply(calc_log_return)
        .reset_index(level=0, drop=True)
    )

    # Multiple lookback windows
    for w in [2, 5, 10, 30, 60]:
        col_name = f"log_ret_{w}m"
        df[col_name] = (
            df.groupby("symbol")["mid"]
            .apply(lambda x: np.log(x / x.shift(w)))
            .reset_index(level=0, drop=True)
        )

    print("Returns calculation complete")
    return df

def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various moving averages."""
    print("Starting moving averages calculation...")
    df["ewma_spread_10"] = (
        df.groupby("symbol")["spread"]
        .apply(lambda x: x.ewm(span=10, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )
    df["ewma_imbalance_10"] = (
        df.groupby("symbol")["imbalance"]
        .apply(lambda x: x.ewm(span=10, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )

    # Rolling volatility
    for vw in [5, 10, 30, 60]:
        col_name = f"roll_vol_{vw}m"
        df[col_name] = (
            df.groupby("symbol")["log_ret_1m"]
            .apply(lambda x: x.rolling(window=vw, min_periods=vw).std())
            .reset_index(level=0, drop=True)
        )

    print("Moving averages calculation complete")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for volatility prediction."""
    from time import time
    start_total = time()
    print("Starting feature engineering...")
    
    df = df.copy()
    
    # Validation check
    start = time()
    start_row = np.random.randint(0, len(df) - 100)
    sample_df = df.iloc[start_row : start_row + 100]
    if not sample_df[["symbol", "date", "minute"]].equals(
        sample_df.sort_values(["symbol", "date", "minute"])[["symbol", "date", "minute"]]
    ):
        raise ValueError("DataFrame must be sorted by symbol, date, and minute (sample failed)")
    print(f"Validation check completed in {time() - start:.2f} seconds")

    # 1. Time-based features
    start = time()
    df['minute_of_day'] = (
        pd.to_datetime(df['minute'], format='%H:%M').dt.hour * 60
        + pd.to_datetime(df['minute'], format='%H:%M').dt.minute
    )
    df['normalized_time'] = (df['minute_of_day'] - 570) / (960 - 570)
    df['time_cos'] = np.cos(2 * np.pi * df['normalized_time'])
    df['time_sin'] = np.sin(2 * np.pi * df['normalized_time'])
    df['is_morning'] = df['minute_of_day'] < 690
    df['is_lunch'] = (df['minute_of_day'] >= 690) & (df['minute_of_day'] <= 810)
    df['is_afternoon'] = df['minute_of_day'] > 810
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    print(f"Time-based features completed in {time() - start:.2f} seconds")

    # 2. Price action features
    start = time()
    for window in [5, 10, 30, 60]:
        print(f"Processing window {window}m...")
        window_start = time()
        # Rolling high-low range
        df[f'range_{window}m'] = (
            df.groupby('symbol')['mid']
              .apply(lambda x: (
                  x.rolling(window, min_periods=window).max()
                  - x.rolling(window, min_periods=window).min()
              ) / x)
              .reset_index(level=0, drop=True)
        )

        # Rolling volatility of volatility
        df[f'vol_of_vol_{window}m'] = (
            df.groupby('symbol')[f'roll_vol_{window}m']
              .rolling(window, min_periods=window//4)
              .std()
              .reset_index(level=0, drop=True)
        )

        # Price acceleration
        df[f'price_accel_{window}m'] = (
            df.groupby('symbol')['log_ret_1m']
              .apply(lambda x: x.diff().rolling(window, min_periods=window).mean())
              .reset_index(level=0, drop=True)
        )
        print(f"Window {window}m completed in {time() - window_start:.2f} seconds")
    print(f"Price action features completed in {time() - start:.2f} seconds")

    # 3. Microstructure features
    start = time()
    df['spread_ma_10m'] = (
        df.groupby('symbol')['spread']
          .rolling(10, min_periods=5)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df['spread_vol'] = (
        df.groupby('symbol')['spread']
          .rolling(10, min_periods=5)
          .std()
          .reset_index(level=0, drop=True)
    )
    df['imbalance_ma_10m'] = (
        df.groupby('symbol')['imbalance']
          .rolling(10, min_periods=5)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df['imbalance_vol'] = (
        df.groupby('symbol')['imbalance']
          .rolling(10, min_periods=5)
          .std()
          .reset_index(level=0, drop=True)
    )
    print(f"Microstructure features completed in {time() - start:.2f} seconds")

    # 4. Technical features
    start = time()
    def calc_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # RSI calculation
    df['rsi'] = (
        df.groupby('symbol')['mid']
          .apply(calc_rsi, window=14)
          .reset_index(level=0, drop=True)
    )

    # Bollinger Bands
    df['bb_ma'] = (
        df.groupby('symbol')['mid']
          .rolling(window=20)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df['bb_std'] = (
        df.groupby('symbol')['mid']
          .rolling(window=20)
          .std()
          .reset_index(level=0, drop=True)
    )
    df['bb_width'] = df['bb_std'] * 4 / df['bb_ma']
    print(f"Technical features completed in {time() - start:.2f} seconds")

    # 5. Statistical features
    start = time()
    for window in [10, 30, 60]:
        df[f'skew_{window}m'] = (
            df.groupby('symbol')['log_ret_1m']
            .rolling(window, min_periods=window)
            .skew()
            .reset_index(level=0, drop=True)
        )
        df[f'kurt_{window}m'] = (
            df.groupby('symbol')['log_ret_1m']
            .rolling(window, min_periods=window)
            .kurt()
            .reset_index(level=0, drop=True)
        )
    print(f"Statistical features completed in {time() - start:.2f} seconds")

    print(f"Total feature engineering completed in {time() - start_total:.2f} seconds")
    return df

def create_prediction_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create forward-looking volatility targets for prediction."""
    df = df.copy()

    horizons = [5, 10, 30, 60]
    for horizon in horizons:
        target_col = f'target_vol_{horizon}m'
        df[target_col] = (
            df.groupby('symbol')['log_ret_1m']
              .rolling(horizon, min_periods=horizon)
              .std()
              .shift(-horizon)
              .reset_index(level=0, drop=True)
        )

    for horizon in horizons:
        target_col = f'target_vol_{horizon}m'
        mask_col = f'valid_target_{horizon}m'
        df[mask_col] = df[target_col].notna().astype(int)
        df[target_col] = df[target_col].fillna(0.0)

    return df

def print_sanity_check(df: pd.DataFrame):
    """Print sanity check information about the DataFrame."""
    print("\n=== SANITY CHECK ===")
    print(f"Total rows: {len(df):,}")
    print(f"Unique symbols: {df['symbol'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Time range: {df['minute'].min()} to {df['minute'].max()}")

    print("\nSample of data:")
    pd.set_option('display.max_columns', 10)
    print(df.head().to_string())

    print("\nColumn list:")
    for col in sorted(df.columns):
        print(f"- {col}")

    print("\nMemory usage:")
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total memory: {memory_usage:.2f} MB")

    print("\nMissing values (sorted by percentage):")
    total_rows = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total_rows * 100).round(2)
    missing_stats = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Percentage', ascending=False)
    
    # Only show columns that have missing values
    missing_stats = missing_stats[missing_stats['Missing Count'] > 0]
    if not missing_stats.empty:
        print(missing_stats)
    else:
        print("No missing values found in any column")

    print("\nValue ranges for key columns:")
    numeric_cols = ['mid', 'spread', 'imbalance', 'log_turnover']
    for col in numeric_cols:
        if col in df.columns:
            stats = df[col].describe()
            print(f"\n{col}:")
            print(f"  min: {stats['min']:.6f}")
            print(f"  max: {stats['max']:.6f}")
            print(f"  mean: {stats['mean']:.6f}")
            print(f"  std: {stats['std']:.6f}")

    print("\n=== END SANITY CHECK ===\n")


# --------------------------------------------------------
# REFAC: SINGLE-DATE PROCESSING
# --------------------------------------------------------

def process_single_date(dt: str, check: bool = False):
    """
    Run the entire pipeline (load, merge, fill, feature-engineer, etc.)
    for a single date dt in YYYYMMDD format. If check=True, prints a sanity check at the end.
    """
    df = load_and_merge_data(dt)
    df = preprocess_data(df)
    df = fill_minute_data(df)
    df = filter_trading_hours(df)
    
    df = calculate_market_metrics(df)
    
    print("Dropping unnecessary columns...")
    columns_to_drop = [
        "rtype", "publisher_id", 
        "instrument_id", "flags", "sequence", "avg_turnover"
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    print("Reordering columns to keep rank, num_days, etc.")
    desired_order = [
        "date", "minute", "symbol", "side", "price", "size",
        "mid", "spread", "imbalance", "bid_px_00", "ask_px_00",
        "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00",
        "log_turnover", "rank", "num_days", "in_top1000_universe"
    ]
    existing_cols = [c for c in desired_order if c in df.columns]
    df = df[existing_cols + [c for c in df.columns if c not in existing_cols]]
    
    df = calculate_returns(df)
    df = calculate_moving_averages(df)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Creating prediction targets...")
    df = create_prediction_targets(df)
    
    if check:
        print_sanity_check(df)

    print("Saving processed data temporarily and uploading to S3...")
    temp_dir = "temp_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = f"{temp_dir}/{dt}_processed.parquet"
    
    # Save locally
    df.to_parquet(temp_file)
    
    # Upload to S3
    s3 = boto3.client("s3")
    s3_key = f"data/features/itch/{dt}.itch_features.parquet"
    
    try:
        s3.upload_file(temp_file, BUCKET_NAME, s3_key)
        print(f"Successfully uploaded to s3://{BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                # Directory not empty or other issue
                pass
    
    print(f"Processing complete for {dt}!")

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='Process market data for volatility prediction')

    # Make --date optional, but add --all and --only-sanity-check
    parser.add_argument('-d', '--date', type=str, help='Date in YYYYMMDD format')
    parser.add_argument('-a', '--all', action='store_true', help='Process all dates from daily turnover')
    parser.add_argument('-c', '--check', action='store_true', help='Print sanity check information')
    parser.add_argument('-o', '--only-sanity-check', action='store_true', 
                       help='Only perform sanity check on existing file')
    args = parser.parse_args()

    # Handle only-sanity-check mode first
    if args.only_sanity_check:
        if not args.date:
            parser.error("--only-sanity-check requires --date YYYYMMDD")
        
        # Validate date format
        try:
            datetime.strptime(args.date, '%Y%m%d')
        except ValueError:
            raise ValueError("Incorrect date format. Please use YYYYMMDD format (e.g., 20250115)")
        

        s3_key = f"data/features/itch/{args.date}.itch_features.parquet"
        
        if not parquet_exists_in_s3(BUCKET_NAME, args.date):
            print(f"Error: File not found: s3://{BUCKET_NAME}/{s3_key}")
            sys.exit(1)
            
        print(f"Loading existing file from s3://{BUCKET_NAME}/{s3_key} for sanity check...")
        s3 = boto3.client('s3')
        
        try:
            # Create a temporary directory
            temp_dir = "temp_sanity_check"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = f"{temp_dir}/{args.date}_check.parquet"
            
            # Download the file
            s3.download_file(BUCKET_NAME, s3_key, temp_file)
            
            # Read and perform sanity check
            df = pd.read_parquet(temp_file)
            print_sanity_check(df)
            
        except Exception as e:
            print(f"Error during sanity check: {str(e)}")
            raise
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
        
        sys.exit(0)

    # If --all is not set, we require a single date
    if not args.all:
        if not args.date:
            parser.error("Either provide --date YYYYMMDD or use --all to process all available dates.")
        
        # Single-date mode
        # Validate date format
        try:
            datetime.strptime(args.date, '%Y%m%d')
        except ValueError:
            raise ValueError("Incorrect date format. Please use YYYYMMDD format (e.g., 20250115)")

        print(f"Starting processing for single date: {args.date}")
        process_single_date(args.date, check=args.check)

    else:
        # --all mode
        print("Processing ALL dates from S3 daily_turnover.")
        prefix = "data/daily_turnover/"

        all_dates = get_dates_from_s3(BUCKET_NAME, prefix)
        for dt in all_dates:
            if parquet_exists_in_s3(BUCKET_NAME, dt):
                print(f"Already have s3://{BUCKET_NAME}/data/features/itch/{dt}.itch_features.parquet ... Skipping.")
            else:
                print(f"No parquet found for {dt}. Generating now...")
                try:
                    process_single_date(dt, check=args.check)
                except Exception as e:
                    print(f"Error while processing {dt}: {e}")
                    # Decide whether to continue or break; here we continue
                    continue

    print("All requested processing complete!")


if __name__ == "__main__":
    print("Script starting...")
    try:
        main()
    except SystemExit as e:
        # This catches argparse's system exit if arguments are missing
        if str(e) != '0':  # '0' means clean exit
            print("Error: usage: python feature_engineering.py [-d YYYYMMDD] [--all]")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
