import math
import numpy as np
import pandas as pd
from datetime import datetime
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from feature_engineering import get_dates_from_s3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, 
                       message='datetime.datetime.utcnow.*is deprecated')

def merge_vol_data(
    date_str,
    s3_bucket='bam-volatility-project',
    intraday_prefix='data/features/itch/',
    lookback_prefix='data/features/intraday_volatility/',
    neighbors_prefix='data/features/nn/',
    output_prefix='data/features/attention_df/all/',
    neighbor_vol_cols=('roll_vol_5m','roll_vol_10m','roll_vol_30m','roll_vol_60m'),
    neighbor_ret_cols=('log_ret_1m','log_ret_2m','log_ret_5m','log_ret_10m','log_ret_30m','log_ret_60m'),
    special_open_fix=True
):
    """
    Merge intraday features, daily lookback volatility, neighbor-based features,
    and then compute forward 10-minute vol (plus a 1-min-lag variant).
    Ensures minute_offset=380 (15:50) gets a non-NA label if data is available
    for offset=381..389.

    Parameters
    ----------
    date_str : str
        The date in 'YYYYMMDD' format (e.g. '20240116').
    s3_bucket : str
        S3 bucket name where data is located.
    intraday_prefix : str
        S3 prefix/path for the intraday (feature_engineering) output.
    lookback_prefix : str
        S3 prefix/path for the daily lookback volatility output.
    neighbors_prefix : str
        S3 prefix/path for the monthly neighbors output.
    output_prefix : str
        S3 prefix/path where final merged Parquet should be written.
    neighbor_vol_cols : tuple
        Volatility columns for neighbor averaging (must exist in your intraday DataFrame).
    neighbor_ret_cols : tuple
        Return columns for neighbor averaging (must exist in your intraday DataFrame).
    special_open_fix : bool
        Whether to apply special approximations for earliest minutes (9:35..9:39).

    Returns
    -------
    pd.DataFrame
        The merged DataFrame (also written to S3).
    """
    # Convert YYYYMMDD into a datetime
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    fs = s3fs.S3FileSystem(anon=False)

    # ---------------------------------------------------------------------
    # 1) Load Intraday Features
    # ---------------------------------------------------------------------
    intraday_file = f"s3://{s3_bucket}/{intraday_prefix}{date_str}.itch_features.parquet"
    print(f"Loading intraday data from {intraday_file} ...")
    with fs.open(intraday_file, 'rb') as f:
        intraday_df = pd.read_parquet(f)
        
    # Remove all columns starting with 'target_'
    target_cols = [col for col in intraday_df.columns if col.startswith('target_')]
    intraday_df = intraday_df.drop(columns=target_cols)

    # ---------------------------------------------------------------------
    # 2) Filter time range => 9:34..16:00 so we can compute forward vol at 15:50
    # ---------------------------------------------------------------------
    def parse_minute_str(m_str):
        """Parses 'HH:MM' into an integer offset from 9:30=0."""
        hh, mm = map(int, m_str.split(':'))
        return (hh - 9)*60 + (mm - 30)  # 9:30 => offset=0

    intraday_df['minute_offset'] = intraday_df['minute'].apply(parse_minute_str)

    # Keep up to offset=390 (16:00) so we have returns for computing the forward vol at 15:50
    intraday_df = intraday_df[
        (intraday_df['minute_offset'] >= 4) &   # >= 9:34
        (intraday_df['minute_offset'] <= 390)   # <= 16:00 (not just 15:50)
    ].copy()

    # ---------------------------------------------------------------------
    # 3) Merge Daily Lookback Volatility
    # ---------------------------------------------------------------------
    lookback_file = f"s3://{s3_bucket}/{lookback_prefix}{date_str}.average_intraday_volatility.parquet"
    print(f"Loading daily lookback from {lookback_file} ...")
    with fs.open(lookback_file, 'rb') as f:
        lookback_df = pd.read_parquet(f)
        # Drop the date column if it exists in the lookback data
        if 'date' in lookback_df.columns:
            lookback_df = lookback_df.drop(columns=['date'])
        
        print("\nLookback columns before merge:", lookback_df.columns.tolist())
        print("Sample of lookback data:")
        print(lookback_df.head().to_string())

    # Merge only on symbol since each file represents one date
    intraday_df = pd.merge(
        intraday_df,
        lookback_df,
        on='symbol',
        how='left'
    )
    
    print("\nColumns after lookback merge:", intraday_df.columns.tolist())
    print("Sample after merge:")
    print(intraday_df[['symbol', 'minute_offset', 'avg_vol_1d', 'avg_vol_2d', 'avg_vol_5d', 'avg_vol_10d']].head().to_string())

    # ---------------------------------------------------------------------
    # 4) Load Monthly Neighbors => symbol2neighbors
    # ---------------------------------------------------------------------
    month_str = date_obj.strftime("%Y%m")
    neighbor_file = f"s3://{s3_bucket}/{neighbors_prefix}{month_str}_neighbors.csv"
    print(f"Loading monthly neighbors from {neighbor_file} ...")
    with fs.open(neighbor_file, 'rb') as f:
        neigh_df = pd.read_csv(f)

    neighbor_cols = [c for c in neigh_df.columns if c.startswith('n')]
    symbol2neighbors = {
        row['symbol']: [row[c] for c in neighbor_cols if pd.notna(row[c])]
        for _, row in neigh_df.iterrows()
    }

    # ---------------------------------------------------------------------
    # 5) Build neighbor_data for each minute_offset - MEMORY OPTIMIZATION
    # ---------------------------------------------------------------------
    keep_cols = ['symbol','minute_offset'] + list(neighbor_vol_cols) + list(neighbor_ret_cols)
    missing_cols = set(keep_cols) - set(intraday_df.columns)
    if missing_cols:
        raise ValueError(f"These neighbor columns are missing: {missing_cols}")

    # Process neighbors in chunks to reduce memory usage
    chunk_size = 25  # Adjust this based on your memory constraints
    unique_symbols = intraday_df['symbol'].unique()
    neighbor_agg_chunks = []
    
    for i in range(0, len(unique_symbols), chunk_size):
        symbol_chunk = unique_symbols[i:i + chunk_size]
        
        # Filter data for current chunk of symbols
        chunk_data = intraday_df[intraday_df['symbol'].isin(symbol_chunk)][keep_cols].copy()
        chunk_data = chunk_data.rename(columns={'symbol': 'neighbor'})
        
        # Build symbol-neighbor pairs for current chunk
        sym_neigh_rows = []
        for s in symbol_chunk:
            neighs = symbol2neighbors.get(s, [])
            for n in neighs:
                sym_neigh_rows.append((s, n))
        
        if not sym_neigh_rows:  # Skip if no neighbors found
            continue
            
        sym_neigh_df = pd.DataFrame(sym_neigh_rows, columns=['symbol','neighbor'])
        base_expand = intraday_df[
            intraday_df['symbol'].isin(symbol_chunk)
        ][['symbol','minute_offset']].drop_duplicates()
        
        expanded_df = pd.merge(base_expand, sym_neigh_df, on='symbol', how='inner')
        expanded_df = pd.merge(
            expanded_df,
            chunk_data,
            left_on=['neighbor','minute_offset'],
            right_on=['neighbor','minute_offset'],
            how='left'
        )
        
        # Aggregate for current chunk
        agg_map = {col: 'mean' for col in neighbor_vol_cols + neighbor_ret_cols}
        chunk_agg = expanded_df.groupby(
            ['symbol','minute_offset'], 
            as_index=False
        ).agg(agg_map)
        
        neighbor_agg_chunks.append(chunk_agg)
        
        # Clean up to free memory
        del expanded_df, chunk_data, sym_neigh_df, base_expand
        
    # Combine all chunks
    neighbor_agg = pd.concat(neighbor_agg_chunks, ignore_index=True)
    del neighbor_agg_chunks  # Clean up

    # Rename columns
    rename_dict = {
        **{c: f'avg_nn_{c}' for c in neighbor_vol_cols},
        **{c: f'avg_nn_{c}' for c in neighbor_ret_cols}
    }
    neighbor_agg.rename(columns=rename_dict, inplace=True)

    # Final merge
    intraday_df = pd.merge(
        intraday_df,
        neighbor_agg,
        on=['symbol','minute_offset'],
        how='left'
    )
    del neighbor_agg  # Clean up

    # ---------------------------------------------------------------------
    # 7) Special open fix for 9:35..9:39 => roll_vol_10m = sqrt(2)*roll_vol_5m
    # ---------------------------------------------------------------------
    if special_open_fix and 'roll_vol_10m' in intraday_df.columns and 'roll_vol_5m' in intraday_df.columns:
        cond_935_939 = intraday_df['minute_offset'].between(5, 9)
        intraday_df.loc[cond_935_939, 'roll_vol_10m'] = (
            np.sqrt(2) * intraday_df.loc[cond_935_939, 'roll_vol_5m']
        )
    # ---------------------------------------------------------------------
    # 8) Compute forward 10-min volatility => 'Y_log_vol_10min' (includes minute i)
    #    Then compute a 1-minute-lag => 'Y_log_vol_10min_lag_1m'
    #    Also compute forward 60-min return => 'Y_log_ret_60min' and its lag
    # ---------------------------------------------------------------------
    intraday_df = intraday_df.sort_values(["symbol","minute_offset"])
    group_cols = ["symbol"]

    def compute_forward_metrics(df_group):
        df_group = df_group.reset_index(drop=True)
        rets = df_group["log_ret_1m"].values
        n = len(rets)
        fwd_vol = np.full(n, np.nan)
        fwd_ret_60m = np.full(n, np.nan)
        fwd_ret_30m = np.full(n, np.nan)  # Adding 30m for intermediate horizon
        fwd_ret_10m = np.full(n, np.nan)  # Adding 10m to match volatility window

        for i in range(n):
            # 1) Forward volatility (10-minute window)
            end = min(i + 10, n)
            window = rets[i:end]
            valid_points = window[~np.isnan(window)]
            if len(valid_points) >= 2:
                fwd_vol[i] = np.std(valid_points, ddof=1)
                # Also compute 10m return while we have this window
                if not np.any(np.isnan(window)):
                    fwd_ret_10m[i] = np.sum(window)

            # 2) Forward 30-minute return (partial returns for edge cases)
            end_30 = min(i + 30, n)
            if end_30 > i:  # If we have at least 1 minute forward
                window_30 = rets[i:end_30]
                if not np.any(np.isnan(window_30)):
                    fwd_ret_30m[i] = np.sum(window_30)
                
            # 3) Forward 60-minute return (partial returns for edge cases)
            end_60 = min(i + 60, n)
            if end_60 > i:  # If we have at least 1 minute forward
                window_60 = rets[i:end_60]
                if not np.any(np.isnan(window_60)):
                    fwd_ret_60m[i] = np.sum(window_60)
        
        # Add all metrics to dataframe with new names
        df_group["Y_log_vol_10min"] = fwd_vol
        df_group["Y_log_ret_10min"] = fwd_ret_10m
        df_group["Y_log_ret_30min"] = fwd_ret_30m
        df_group["Y_log_ret_60min"] = fwd_ret_60m
        
        # Update partial flags to match new names
        df_group["Y_log_ret_10min_partial"] = df_group["minute_offset"] > 370  # Last 10 minutes
        df_group["Y_log_ret_30min_partial"] = df_group["minute_offset"] > 350  # Last 30 minutes
        df_group["Y_log_ret_60min_partial"] = df_group["minute_offset"] > 320  # Last 60 minutes
        
        # Create 1-minute-lag versions with new names
        for col, source in [
            ("Y_log_vol_10min", fwd_vol),
            ("Y_log_ret_10min", fwd_ret_10m),
            ("Y_log_ret_30min", fwd_ret_30m),
            ("Y_log_ret_60min", fwd_ret_60m)
        ]:
            lagged = np.roll(source, -1)
            lagged[-1] = np.nan
            df_group[f"{col}_lag_1m"] = lagged

        return df_group

    intraday_df = intraday_df.groupby(group_cols, group_keys=False).apply(compute_forward_metrics)

    # ---------------------------------------------------------------------
    # 9) Now that we have Y_log_vol_10min, we filter final offsets <= 380
    #     => This means offset=381..389 won't appear in the final dataset,
    #        but they WERE used to compute Y_log_vol_10min at offset=380.
    # ---------------------------------------------------------------------
    intraday_df = intraday_df[intraday_df["minute_offset"] <= 380].copy()

    # ---------------------------------------------------------------------
    # 10) Add num_missing_features
    # ---------------------------------------------------------------------
    exclude_cols = ['symbol','minute','minute_offset','in_top1000_universe']
    all_cols = intraday_df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in exclude_cols]
    intraday_df['num_missing_features'] = intraday_df[feature_cols].isna().sum(axis=1)

    # ---------------------------------------------------------------------
    # 11) Final filter => in_top1000_universe == True
    # ---------------------------------------------------------------------
    final_df = intraday_df[intraday_df['in_top1000_universe'] == True].copy()

    # ---------------------------------------------------------------------
    # 12) Write final parquet to S3, then sanity-check read it
    # ---------------------------------------------------------------------
    output_file = f"s3://{s3_bucket}/{output_prefix}{date_str}.parquet"
    print(f"Writing final merged dataset to {output_file} ...")
    table = pa.Table.from_pandas(final_df)
    with fs.open(output_file, 'wb') as f:
        pq.write_table(table, f, compression='snappy')

    # Quick read-back check
    with fs.open(output_file, 'rb') as f:
        check_df = pd.read_parquet(f)

    # Final cleanup and column ordering
    # ---------------------------------------------------------------------
    # Remove date column and reorder columns
    check_df = check_df.drop(columns=['date'])
    
    # Separate columns into categories
    starting_cols = ['minute', 'symbol', 'rank', 'num_days', 'in_top1000_universe']
    other_cols = [col for col in check_df.columns if col not in starting_cols 
                  and not col.startswith('valid_') and not col.startswith('Y_')]
    valid_cols = [col for col in check_df.columns if col.startswith('valid_')]
    y_cols = [col for col in check_df.columns if col.startswith('Y_')]
    
    # Sort all but the first two columns
    other_cols.sort()
    valid_cols.sort()
    y_cols.sort()
    
    # Reorder columns
    final_column_order = starting_cols + other_cols + valid_cols + y_cols
    check_df = check_df[final_column_order]
    
    # Create sanity check report
    sanity_check = []
    sanity_check.append("\n" + "="*80)
    sanity_check.append("=== Final Dataset Summary ===")
    sanity_check.append("="*80)
    sanity_check.append(f"Final row count: {len(check_df):,}")
    sanity_check.append(f"Number of columns: {len(check_df.columns)}")
    
    # Print columns in groups
    sanity_check.append("\n=== Column Groups ===")
    sanity_check.append("\n1. Index Columns:")
    for i, col in enumerate(starting_cols, 1):
        sanity_check.append(f"   {i:2d}. {col}")
    
    sanity_check.append("\n2. Feature Columns:")
    for i, col in enumerate(other_cols, 1):
        sanity_check.append(f"   {i:2d}. {col}")
    
    sanity_check.append("\n3. Validation Flags:")
    for i, col in enumerate(valid_cols, 1):
        sanity_check.append(f"   {i:2d}. {col}")
    
    sanity_check.append("\n4. Target Variables:")
    for i, col in enumerate(y_cols, 1):
        sanity_check.append(f"   {i:2d}. {col}")
    
    # Check for NA columns
    na_cols = [col for col in check_df.columns if check_df[col].isna().all()]
    if na_cols:
        sanity_check.append("\n!!! WARNING: THE FOLLOWING COLUMNS CONTAIN ALL NA VALUES !!!")
        for col in na_cols:
            sanity_check.append(f"  - {col}")
    
    print("\n".join(sanity_check))

    return final_df


def process_date_range(start_date, end_date, overwrite=False, **kwargs):
    """Process all dates between start_date and end_date inclusive."""
    from datetime import datetime

    all_dates = get_dates_from_s3(
        kwargs.get('s3_bucket', 'bam-volatility-project'),
        kwargs.get('intraday_prefix', 'data/features/itch/')
    )
    
    # Convert input dates to datetime for comparison
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    # Filter dates within range
    dates_to_process = [
        d for d in all_dates 
        if start_dt <= datetime.strptime(d, "%Y%m%d") <= end_dt
    ]
    
    if not dates_to_process:
        raise ValueError(f"No valid dates found between {start_date} and {end_date}")
    
    print(f"\nProcessing {len(dates_to_process)} dates from {dates_to_process[0]} to {dates_to_process[-1]}")
    
    fs = s3fs.S3FileSystem(anon=False)
    s3_bucket = kwargs.get('s3_bucket', 'bam-volatility-project')
    output_prefix = kwargs.get('output_prefix', 'data/features/attention_df/all')
    
    for date in dates_to_process:
        output_file = f"s3://{s3_bucket}/{output_prefix}{date}.parquet"
        
        # Check if file exists
        if not overwrite and fs.exists(output_file):
            print(f"\nSkipping {date} - output file already exists.")
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing {date}")
        print('='*80)
        try:
            merge_vol_data(date, **kwargs)
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge volatility data for a date range.')
    parser.add_argument('-s', '--start-date', required=True,
                       help='Start date in YYYYMMDD format (e.g., 20240116).')
    parser.add_argument('-e', '--end-date', required=True,
                       help='End date in YYYYMMDD format (e.g., 20240131).')
    parser.add_argument('-o', '--overwrite', action='store_true',
                       help='Overwrite existing output files (default: False).')
    
    args = parser.parse_args()
    
    # Validate date formats
    try:
        start_dt = datetime.strptime(args.start_date, "%Y%m%d")
        end_dt = datetime.strptime(args.end_date, "%Y%m%d")
        if end_dt < start_dt:
            raise ValueError("End date must be >= start date.")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {str(e)}. Must be YYYYMMDD (e.g., 20240116).")
    
    process_date_range(args.start_date, args.end_date, overwrite=args.overwrite)
