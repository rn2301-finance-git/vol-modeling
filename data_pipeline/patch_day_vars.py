import pandas as pd
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from datetime import datetime
from feature_engineering import get_dates_from_s3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def add_daily_features(df: pd.DataFrame, short_window: int = 20, long_window: int = 40):
    """
    Add daily regime features with a validity mask.

    Parameters:
    - df: DataFrame with 'avg_vol_*d' columns (note: there is no 'date' column).
    - short_window: Short lookback window (default 20 days).
    - long_window: Long lookback window (default 40 days).

    The validity flag is now derived from whether the long-window volatility data is available.
    If a symbol lacks sufficient lookback history (i.e. its avg_vol_{long_window}d is NA),
    then the regime is marked as invalid.
    """
    # Get first value per symbol for daily metrics
    daily_metrics = (df.groupby('symbol')
                      .agg({
                          f'avg_vol_{short_window}d': 'first',
                          f'avg_vol_{long_window}d': 'first'
                      })
                      .reset_index())
    
    # Calculate market averages
    market_vol_short = daily_metrics[f'avg_vol_{short_window}d'].mean()
    market_vol_long = daily_metrics[f'avg_vol_{long_window}d'].mean()
    
    # Add relative metrics
    daily_metrics['rel_vol'] = daily_metrics[f'avg_vol_{short_window}d'] / market_vol_short
    daily_metrics['vol_regime'] = market_vol_short / market_vol_long
    daily_metrics['vol_rank'] = daily_metrics[f'avg_vol_{short_window}d'].rank(pct=True)
    
    # Merge with original dataframe
    df = pd.merge(df, daily_metrics[['symbol', 'rel_vol', 'vol_regime', 'vol_rank']], 
                  on='symbol', how='left')
    
    # ---------------------------------------------------------------------
    # Modify validity flag: since the daily df does not have a 'date' column,
    # use the availability of lookback volatility data instead.
    # If the long-window lookback volatility is missing, mark regime_valid as 0.
    # ---------------------------------------------------------------------
    df['regime_valid'] = (~df[f'avg_vol_{long_window}d'].isna()).astype(int)
    
    # Keep regime = 1.0 (neutral) when invalid
    df.loc[df['regime_valid'] == 0, ['vol_regime', 'rel_vol', 'vol_rank']] = 1.0
    
    # Get current column order
    cols = df.columns.tolist()
    
    # Find insertion point (after avg_vol_* columns)
    vol_cols = [col for col in cols if col.startswith('avg_vol_')]
    if vol_cols:
        last_vol_idx = cols.index(sorted(vol_cols)[-1])
        insert_idx = last_vol_idx + 1
    else:
        # If no avg_vol columns, insert after basic columns
        basic_cols = ['minute', 'symbol', 'rank', 'num_days', 'in_top1000_universe']
        insert_idx = len(basic_cols)
    
    # Reorder columns to insert new features at the right position
    new_cols = ['rel_vol', 'vol_regime', 'vol_rank']
    remaining_cols = [col for col in cols if col not in new_cols]
    
    final_cols = (remaining_cols[:insert_idx] + 
                  new_cols + 
                  remaining_cols[insert_idx:])
    
    return df[final_cols]

def process_single_date(date_str: str, 
                        s3_bucket: str = 'volatility-project',
                        input_prefix: str = 'data/features/attention_df/all/',
                        output_prefix: str = 'data/features/attention_df/all/',
                        short_window: int = 5,
                        long_window: int = 10) -> None:
    """Process a single date's parquet file."""
    fs = s3fs.S3FileSystem(anon=False)
    input_file = f"s3://{s3_bucket}/{input_prefix}{date_str}.parquet"
    output_file = f"s3://{s3_bucket}/{output_prefix}{date_str}.parquet"
    
    print(f"Processing {date_str}...")
    
    # Read parquet file
    with fs.open(input_file, 'rb') as f:
        df = pd.read_parquet(f)
    
    # Store original shape and columns for sanity checking
    original_rows = df.shape[0]
    original_columns = set(df.columns)
    
    # Add daily features
    df_patched = add_daily_features(df, short_window, long_window)
    
    # Sanity checks: assert row count is preserved and all original columns exist
    assert df_patched.shape[0] == original_rows, (
        f"Row count mismatch: original {original_rows} vs patched {df_patched.shape[0]}"
    )
    
    missing_cols = original_columns - set(df_patched.columns)
    assert not missing_cols, (
        f"The following original columns are missing in the patched DataFrame: {missing_cols}"
    )
    
    print("Sanity check passed: row count and original columns are preserved.")
    print("\nSample of the first 5 rows:")
    print(df_patched.head(5).to_string())
    print("\nSample of the last 5 rows:")
    print(df_patched.tail(5).to_string())
    if original_rows >= 10:
        print("\nSample of 5 random rows:")
        print(df_patched.sample(5, random_state=42).to_string())
    
    # Write back to S3
    table = pa.Table.from_pandas(df_patched)
    with fs.open(output_file, 'wb') as f:
        pq.write_table(table, f, compression='snappy')
    
    print(f"Successfully processed {date_str}")

def main():
    parser = argparse.ArgumentParser(description='Patch daily variables in parquet files.')
    parser.add_argument('-s', '--start-date', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('-b', '--bucket', default='volatility-project', help='S3 bucket name')
    parser.add_argument('-i', '--input-prefix', default='data/features/attention_df/all/',
                        help='Input S3 prefix')
    parser.add_argument('-o', '--output-prefix', default='data/features/attention_df/all/',
                        help='Output S3 prefix')
    parser.add_argument('--short-window', type=int, default=5, help='Short volatility window')
    parser.add_argument('--long-window', type=int, default=10, help='Long volatility window')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Get list of dates to process
    dates = get_dates_from_s3(args.bucket, args.input_prefix)
    dates = [d for d in dates if args.start_date <= d <= args.end_date]
    
    for date in dates:
        try:
            process_single_date(
                date,
                s3_bucket=args.bucket,
                input_prefix=args.input_prefix,
                output_prefix=args.output_prefix,
                short_window=args.short_window,
                long_window=args.long_window
            )
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
