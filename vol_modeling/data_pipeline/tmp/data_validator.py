import boto3
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import argparse

class DataValidator:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def list_available_dates(self, prefix):
        """List all dates for which we have data"""
        available_files = defaultdict(list)
        
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.dbn.zst'):
                    # Extract date from filename (assumes format *YYYYMMDD*.dbn.zst)
                    for part in key.split('/'):
                        if len(part) >= 8:
                            try:
                                date_str = next(p for p in part.split('-') if len(p) == 8 and p.isdigit())
                                date = datetime.strptime(date_str, '%Y%m%d').date()
                                available_files[date].append(key)
                                break
                            except (StopIteration, ValueError):
                                continue
        
        return available_files

    def find_missing_dates(self, start_date, end_date, data_type):
        """Find missing dates in the date range"""
        prefix = f"databento/{data_type}"
        available_files = self.list_available_dates(prefix)
        
        # Convert dates to datetime.date objects if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Generate all dates in range
        date_range = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                date_range.append(current)
            current += timedelta(days=1)
        
        # Find missing dates
        missing_dates = []
        for date in date_range:
            if date not in available_files:
                missing_dates.append(date)
        
        return missing_dates, available_files

    def print_summary(self, start_date, end_date, data_type):
        """Print a summary of data completeness"""
        missing_dates, available_files = self.find_missing_dates(start_date, end_date, data_type)
        
        print(f"\nData Completeness Report for {data_type}")
        print(f"Date Range: {start_date} to {end_date}")
        print("-" * 50)
        
        if missing_dates:
            print("\nMissing Dates:")
            for date in missing_dates:
                print(f"  - {date}")
        else:
            print("\nNo missing dates found!")
            
        print(f"\nTotal available dates: {len(available_files)}")
        print(f"Total missing dates: {len(missing_dates)}")
        
        # Print file counts by date
        print("\nFiles per date:")
        for date in sorted(available_files.keys()):
            files = available_files[date]
            print(f"  {date}: {len(files)} files")

def main():
    parser = argparse.ArgumentParser(description='Validate data completeness')
    parser.add_argument('-b', '--bucket', required=True, help='S3 bucket name')
    parser.add_argument('-t', '--type', choices=['ITCH', 'OHLCV'], required=True,
                      help='Type of data to validate')
    parser.add_argument('-s', '--start-date', required=True, 
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('-e', '--end-date', required=True,
                      help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    validator = DataValidator(args.bucket)
    validator.print_summary(args.start_date, args.end_date, args.type)

if __name__ == "__main__":
    main()