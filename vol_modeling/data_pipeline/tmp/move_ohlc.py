import boto3

def fix_directory_structure(bucket_name):
    s3 = boto3.client('s3')
    
    # Source and destination within same date directory
    base_prefix = "databento/OHLCV/2025/01/15/"
    
    print(f"Moving files into data/ subdirectory in {base_prefix}")
    
    # List files in source directory
    paginator = s3.get_paginator('list_objects_v2')
    moved = 0
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                src_key = obj['Key']
                if not src_key.endswith('.dbn.zst'):  # Skip non-data files
                    continue
                    
                # Only process files that aren't already in data/
                if '/data/' not in src_key:
                    filename = src_key.split('/')[-1]
                    dst_key = f"{base_prefix}data/{filename}"
                    
                    print(f"Moving {filename} to data/ subdirectory...")
                    
                    # Copy to new location
                    s3.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': src_key},
                        Key=dst_key
                    )
                    
                    # Delete from old location
                    s3.delete_object(Bucket=bucket_name, Key=src_key)
                    moved += 1
                
        print(f"\nMoved {moved} files into data/ subdirectory")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    BUCKET_NAME = "volatility-project"
    fix_directory_structure(BUCKET_NAME)