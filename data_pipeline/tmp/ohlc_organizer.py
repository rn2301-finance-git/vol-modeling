import boto3
from datetime import datetime
import pandas as pd

def list_s3_objects(s3, bucket, prefix):
    """List all objects with given prefix"""
    try:
        paginator = s3.get_paginator('list_objects_v2')
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                files.extend(page['Contents'])
        return files
    except Exception as e:
        print(f"Error listing objects with prefix {prefix}: {str(e)}")
        return []

def examine_files(bucket_name):
    """Examine and print raw file information"""
    s3 = boto3.client('s3')
    
    # Define possible prefixes
    prefixes = [
        'databento/OHLCV',
        'raw_data/XNAS',
    ]
    
    for prefix in prefixes:
        print(f"\nExamining prefix: {prefix}")
        print("-" * 50)
        
        files = list_s3_objects(s3, bucket_name, prefix)
        if not files:
            print(f"No files found in {prefix}/")
            continue
            
        print(f"Found {len(files)} files")
        print("\nSample of first 5 files:")
        for obj in sorted(files, key=lambda x: x['Key'])[:5]:
            key = obj['Key']
            size_mb = obj['Size'] / (1024 * 1024)
            print(f"\nKey: {key}")
            print(f"Size: {size_mb:.2f} MB")
            print(f"Last Modified: {obj['LastModified']}")
            
            # Try to break down the filename
            parts = key.split('/')
            if len(parts) > 1:
                print("Path parts:", parts)
                filename = parts[-1]
                print("Filename:", filename)
                
                # Try different date extraction methods
                if '-' in filename:
                    date_parts = filename.split('-')
                    print("Date parts:", date_parts)

def organize_ohlc_data(bucket_name):
    """List and organize OHLC data from different request dates"""
    s3 = boto3.client('s3')
    examine_files(bucket_name)  # First examine the files
    
    print("\nWould you like to proceed with organization? (y/n)")

if __name__ == "__main__":
    BUCKET_NAME = "bam-volatility-project"
    organize_ohlc_data(BUCKET_NAME)