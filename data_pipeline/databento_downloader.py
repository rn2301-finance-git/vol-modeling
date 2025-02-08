from ftp_to_s3 import FTPtoS3Transfer
from datetime import datetime
import argparse
import os

class DabentoDownloader:
    def __init__(self, bucket_name, ftp_user, ftp_pass=None):
        """
        Initialize Databento downloader
        
        Args:
            bucket_name (str): AWS S3 bucket name
            ftp_user (str): Databento FTP username
            ftp_pass (str, optional): Databento FTP password. If None, will check environment variable
        """
        self.bucket_name = bucket_name
        self.ftp_user = ftp_user
        
        if ftp_pass is None:
            ftp_pass = os.environ.get('DATABENTO_PASSWORD')
            if not ftp_pass:
                raise ValueError("FTP password not provided and DATABENTO_PASSWORD environment variable not set")
        
        self.ftp_pass = ftp_pass
        self.ftp_host = "ftp.databento.com"

    def download_dataset(self, request_id, dataset_type, s3_prefix=None):
        """
        Download a specific Databento dataset
        
        Args:
            request_id (str): Databento request ID (e.g., 'XNAS-20250115-J6GFSUD4AV')
            dataset_type (str): Type of data ('ITCH' or 'OHLCV')
            s3_prefix (str, optional): Custom S3 prefix. If None, will be auto-generated
        """
        ftp_path = f"/6RNGNVWH/{request_id}"
        
        if s3_prefix is None:
            s3_prefix = f"databento/{dataset_type}/"

        transfer = FTPtoS3Transfer(
            ftp_host=self.ftp_host,
            ftp_user=self.ftp_user,
            ftp_pass=self.ftp_pass,
            bucket_name=self.bucket_name
        )

        try:
            print(f"Starting transfer for {dataset_type} data...")
            print(f"FTP Path: {ftp_path}")
            print(f"S3 Prefix: {s3_prefix}")
            
            successful, failed = transfer.transfer_files(
                ftp_directory=ftp_path,
                s3_prefix=s3_prefix
            )
            print(f"\nTransfer summary for {request_id}:")
            print(f"Successful transfers: {successful}")
            print(f"Failed transfers: {failed}")
            
            return successful, failed
        finally:
            transfer.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Download Databento datasets to S3')
    parser.add_argument('-r', '--request-id', required=True, help='Databento request ID')
    parser.add_argument('-t', '--dataset-type', choices=['ITCH', 'OHLCV'], required=True, 
                      help='Type of dataset (ITCH or OHLCV)')
    parser.add_argument('-b', '--bucket', required=True, help='S3 bucket name')
    parser.add_argument('-u', '--ftp-user', required=True, help='Databento FTP username')
    parser.add_argument('-p', '--ftp-pass', help='Databento FTP password (or use DATABENTO_PASSWORD env var)')
    parser.add_argument('-s', '--s3-prefix', help='Custom S3 prefix (optional)')
    
    parser.epilog = '''
    Examples:
        # Download ITCH data using environment variable for password
        python %(prog)s -r XNAS-20250115-B4WLX39FJT -t ITCH -b my-bucket -u user@email.com
        
        # Download OHLCV data with custom S3 prefix
        python %(prog)s -r XNAS-20250115-J6GFSUD4AV -t OHLCV -b my-bucket -u user@email.com -s custom/path/
    '''
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    
    args = parser.parse_args()
    
    downloader = DabentoDownloader(
        bucket_name=args.bucket,
        ftp_user=args.ftp_user,
        ftp_pass=args.ftp_pass
    )
    
    downloader.download_dataset(
        request_id=args.request_id,
        dataset_type=args.dataset_type,
        s3_prefix=args.s3_prefix
    )

if __name__ == "__main__":
    main()