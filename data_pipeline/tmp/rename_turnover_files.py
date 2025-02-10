import boto3
import re
from datetime import datetime
import os
def rename_turnover_files(bucket_name: str, prefix: str = "data/daily_turnover/"):
    """
    Rename files from pattern YYYYMMDD_turnover.csv to YYYYMMDD.turnover.csv in S3
    """
    s3_client = boto3.client('s3')
    
    # List all objects in the prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    # Pattern to match YYYYMMDD_turnover.csv
    pattern = r'(\d{8})_turnover\.csv$'
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            old_key = obj['Key']
            match = re.search(pattern, old_key)
            
            if match:
                # Create new key with the desired format
                date_str = match.group(1)
                prefix_path = old_key[:old_key.rfind('/') + 1]
                new_key = f"{prefix_path}{date_str}.turnover.csv"
                
                print(f"Renaming: {old_key} -> {new_key}")
                
                # Copy object to new key
                s3_client.copy_object(
                    Bucket=bucket_name,
                    CopySource={'Bucket': bucket_name, 'Key': old_key},
                    Key=new_key
                )
                
                # Delete old object
                s3_client.delete_object(Bucket=bucket_name, Key=old_key)

if __name__ == "__main__":
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    rename_turnover_files(BUCKET_NAME)
