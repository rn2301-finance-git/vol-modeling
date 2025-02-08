import boto3
import os

def move_to_parent_directory(bucket_name):
    s3 = boto3.client('s3')
    
    # Source and destination prefixes
    source_base = "databento/ITCH/2025/01/15/"
    dest_base = "databento/ITCH/"
    
    print(f"Moving files from {source_base} to {dest_base}")
    
    # List files in source directory
    paginator = s3.get_paginator('list_objects_v2')
    moved = 0
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=source_base):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                src_key = obj['Key']
                
                # Check if file is in data or metadata subdirectory
                if '/data/' in src_key or '/metadata/' in src_key:
                    # Extract the subdirectory (data or metadata) and filename
                    parts = src_key.split('/')
                    subdir = 'data' if '/data/' in src_key else 'metadata'
                    filename = parts[-1]
                    
                    # Construct new key in parent directory
                    dst_key = f"{dest_base}{subdir}/{filename}"
                    
                    print(f"Moving {src_key} to {dst_key}")
                    
                    # Copy to new location
                    s3.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': src_key},
                        Key=dst_key
                    )
                    
                    # Delete from old location
                    s3.delete_object(Bucket=bucket_name, Key=src_key)
                    moved += 1
                
        print(f"\nMoved {moved} files to parent directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    move_to_parent_directory(BUCKET_NAME)