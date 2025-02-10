import boto3
import pandas as pd
import numpy as np
from io import BytesIO
import sys
import os
def download_parquet_to_df(bucket: str, key: str) -> pd.DataFrame:
    """Download a parquet file from S3 and return as DataFrame."""
    print(f"Downloading s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    
    obj = s3.get_object(Bucket=bucket, Key=key)
    parquet_buffer = BytesIO(obj['Body'].read())
    return pd.read_parquet(parquet_buffer)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Compare two DataFrames and print detailed differences."""
    print("\n=== DataFrame Comparison ===")
    
    # Basic info
    print("\nBasic Information:")
    print(f"DataFrame 1 shape: {df1.shape}")
    print(f"DataFrame 2 shape: {df2.shape}")
    
    # Compare columns
    print("\nColumn Comparison:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 != cols2:
        print("Column differences found!")
        print("Columns only in df1:", cols1 - cols2)
        print("Columns only in df2:", cols2 - cols1)
    else:
        print("Both DataFrames have the same columns")
    
    # Compare data types
    print("\nData Type Comparison:")
    dtypes1 = df1.dtypes
    dtypes2 = df2.dtypes
    dtype_diff = dtypes1 != dtypes2
    if dtype_diff.any():
        print("Data type differences found:")
        for col in dtype_diff[dtype_diff].index:
            print(f"{col}: df1={dtypes1[col]} vs df2={dtypes2[col]}")
    else:
        print("All data types match")
    
    # Compare actual data
    common_cols = list(cols1.intersection(cols2))
    print("\nData Value Comparison:")
    
    # Sort both DataFrames the same way
    sort_cols = ['symbol', 'date', 'minute'] if all(col in common_cols for col in ['symbol', 'date', 'minute']) else common_cols[:2]
    df1_sorted = df1.sort_values(sort_cols).reset_index(drop=True)
    df2_sorted = df2.sort_values(sort_cols).reset_index(drop=True)
    
    for col in common_cols:
        if df1[col].dtype in [np.float64, np.float32]:
            # For float columns, check if values are close
            is_different = ~np.isclose(df1_sorted[col], df2_sorted[col], equal_nan=True)
        else:
            # For other types, check exact equality
            is_different = df1_sorted[col] != df2_sorted[col]
        
        if is_different.any():
            diff_count = is_different.sum()
            print(f"\nDifferences in column '{col}':")
            print(f"Number of differences: {diff_count}")
            
            # Show first few differences
            # Convert boolean mask to indices
            diff_indices = np.where(is_different)[0][:5]
            print("First few differences:")
            for idx in diff_indices:
                print(f"Index {idx}:")
                print(f"  df1: {df1_sorted.iloc[idx][col]}")
                print(f"  df2: {df2_sorted.iloc[idx][col]}")
        
def main():
    BUCKET = os.environ.get('BUCKET_NAME')
    KEY1 = "data/features/itch/20240506.itch_features.parquet"
    KEY2 = "data/features/itch/old/20240506.itch_features.parquet"
    
    try:
        # Download and load both files
        df1 = download_parquet_to_df(BUCKET, KEY1)
        df2 = download_parquet_to_df(BUCKET, KEY2)
        
        # Compare the DataFrames
        compare_dataframes(df1, df2)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
