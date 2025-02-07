import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from datetime import datetime

def create_topn_subset(
    date_str,
    n=10,  # New parameter for arbitrary N
    s3_bucket='bam-volatility-project',
    input_prefix='data/features/attention_df/all/',
    output_prefix='data/features/attention_df/top_n/'
):
    """
    Creates a subset of the data containing only the top N stocks by rank.
    
    Parameters
    ----------
    date_str : str
        The date in 'YYYYMMDD' format (e.g. '20240116')
    n : int
        Number of top stocks to include (e.g. 10 for top 10)
    s3_bucket : str
        S3 bucket name where data is located
    input_prefix : str
        S3 prefix/path for the input merged Parquet files
    output_prefix : str
        S3 prefix/path where subset Parquet should be written
    """
    fs = s3fs.S3FileSystem(anon=False)
    
    # Input/output file paths
    input_file = f"s3://{s3_bucket}/{input_prefix}{date_str}.parquet"
    output_file = f"s3://{s3_bucket}/{output_prefix}{date_str}.top{n}.parquet"
    
    print(f"Processing {date_str}...")
    print(f"Reading from {input_file}")
    
    # Read the parquet file
    with fs.open(input_file, 'rb') as f:
        table = pq.read_table(f)
        df = table.to_pandas()
    
    # Filter for top N stocks
    df_subset = df[df['rank'] <= n].copy()
    
    # Write the subset back to S3
    print(f"Writing subset to {output_file}")
    subset_table = pa.Table.from_pandas(df_subset)
    with fs.open(output_file, 'wb') as f:
        pq.write_table(subset_table, f, compression='snappy')
    
    print(f"Completed {date_str}: Original rows: {len(df):,}, Subset rows: {len(df_subset):,}")
    return df_subset

def process_date_range(start_date, end_date, n=10, overwrite=False, **kwargs):
    """Process all dates between start_date and end_date inclusive."""
    # Convert input dates to datetime for comparison
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    fs = s3fs.S3FileSystem(anon=False)
    s3_bucket = kwargs.get('s3_bucket', 'bam-volatility-project')
    input_prefix = kwargs.get('input_prefix', 'data/features/attention_df/all/')
    output_prefix = kwargs.get('output_prefix', 'data/features/attention_df/top_n/')
    # List all files in the input directory
    input_pattern = f"s3://{s3_bucket}/{input_prefix}*.parquet"
    all_files = fs.glob(input_pattern)
    
    # Extract dates from filenames and filter by range
    dates_to_process = []
    for f in all_files:
        date_str = f.split('/')[-1].split('.')[0]
        try:
            file_dt = datetime.strptime(date_str, "%Y%m%d")
            if start_dt <= file_dt <= end_dt:
                dates_to_process.append(date_str)
        except ValueError:
            continue
    
    dates_to_process.sort()
    
    if not dates_to_process:
        raise ValueError(f"No valid dates found between {start_date} and {end_date}")
    
    print(f"\nProcessing {len(dates_to_process)} dates from {dates_to_process[0]} to {dates_to_process[-1]}")
    
    for date in dates_to_process:
        output_file = f"s3://{s3_bucket}/{output_prefix}{date}.top{n}.parquet"
        
        if not overwrite and fs.exists(output_file):
            print(f"\nSkipping {date} - subset file already exists.")
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing {date}")
        print('='*80)
        try:
            create_topn_subset(date, n=n, **kwargs)
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create top N stock subsets from merged data.')
    parser.add_argument('-s', '--start-date', required=True,
                       help='Start date in YYYYMMDD format (e.g., 20240116).')
    parser.add_argument('-e', '--end-date', required=True,
                       help='End date in YYYYMMDD format (e.g., 20240131).')
    parser.add_argument('-n', '--top-n', type=int, default=10,
                       help='Number of top stocks to include (default: 10)')
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
    
    # Validate n
    if args.top_n < 1:
        raise ValueError("top-n must be a positive integer")
    
    process_date_range(args.start_date, args.end_date, n=args.top_n, overwrite=args.overwrite)