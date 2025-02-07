import boto3
import sys

def reorganize_s3_paths(bucket_name, old_prefix, new_prefix):
    """
    Move files from old path structure to new path structure
    
    Args:
        bucket_name (str): S3 bucket name
        old_prefix (str): Old path prefix
        new_prefix (str): New path prefix
    """
    s3 = boto3.client('s3')
    
    print(f"Moving files from {old_prefix} to {new_prefix}")
    print(f"Bucket: {bucket_name}")
    
    # List all objects with the old prefix
    paginator = s3.get_paginator('list_objects_v2')
    total_files = 0
    moved_files = 0
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=old_prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                total_files += 1
                old_key = obj['Key']
                
                # Get the filename from the old path
                filename = old_key.split('/')[-1]
                
                # Construct new key
                new_key = f"{new_prefix}/{filename}"
                
                print(f"\nMoving: {old_key} -> {new_key}")
                
                # Copy object to new location
                try:
                    s3.copy_object(
                        CopySource={'Bucket': bucket_name, 'Key': old_key},
                        Bucket=bucket_name,
                        Key=new_key
                    )
                    
                    # Delete old object
                    s3.delete_object(Bucket=bucket_name, Key=old_key)
                    moved_files += 1
                    print(f"Successfully moved {filename}")
                    
                except Exception as e:
                    print(f"Error moving {filename}: {str(e)}")
        
        print(f"\nOperation complete!")
        print(f"Total files processed: {total_files}")
        print(f"Successfully moved: {moved_files}")
        print(f"Failed: {total_files - moved_files}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = "bam-volatility-project"
    OLD_PREFIX = "raw_data/XNAS/2025-01-15"
    NEW_PREFIX = "databento/ITCH/2025/01/15"
    
    # Execute the move
    reorganize_s3_paths(BUCKET_NAME, OLD_PREFIX, NEW_PREFIX)