import boto3
import re

# S3 bucket and prefix
bucket_name = "volatility-project"
prefix = "data/daily_turnover/"

def get_dates_from_s3(bucket_name, prefix):
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # List objects in the S3 bucket with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        print("No files found in the specified path.")
        return []
    
    # Extract dates from file names
    date_pattern = re.compile(r'(?<!\d)\d{8}(?!\d)')
  # Matches 8-digit dates like 20240116
    dates = set()
    
    for obj in response['Contents']:
        key = obj['Key']
        match = date_pattern.search(key)
        if match:
            dates.add(match.group())
    
    return sorted(dates)

# Call the function
dates = get_dates_from_s3(bucket_name, prefix)
print("Unique dates found:")
print(dates)


