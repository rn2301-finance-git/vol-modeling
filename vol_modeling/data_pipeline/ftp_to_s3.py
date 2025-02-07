import os
import logging
import tempfile
import shutil
from ftplib import FTP
import boto3
from botocore.exceptions import ClientError
import sys

class FTPtoS3Transfer:
    def __init__(self, ftp_host, ftp_user, ftp_pass, bucket_name):
        # Set up logging
        self.logger = logging.getLogger('FTPtoS3Transfer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize FTP connection
        self.ftp = FTP(ftp_host)
        self.ftp.login(user=ftp_user, passwd=ftp_pass)
        self.logger.info(f"Connected to FTP server: {ftp_host}")

        # Initialize S3 client
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.logger.info(f"Initialized S3 client with bucket: {bucket_name}")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temp directory: {self.temp_dir}")

        # Track existing files in S3
        self.existing_files = set()

    def list_s3_files(self, prefix):
        """List all files in the S3 prefix."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    self.existing_files.add(os.path.basename(obj['Key']))
            
            self.logger.info(f"Found {len(self.existing_files)} existing files in S3")
            return self.existing_files
        except ClientError as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return set()

    def file_exists_in_s3(self, filename, s3_prefix):
        """Check if file exists in S3."""
        s3_path = os.path.join(s3_prefix, filename)
        return filename in self.existing_files

    def download_file(self, filename, local_path):
        """Download file from FTP with progress tracking."""
        try:
            # Get file size
            self.ftp.voidcmd('TYPE I')  # Switch to binary mode
            size = self.ftp.size(filename)
            
            with open(local_path, 'wb') as f:
                # Progress tracking
                downloaded = 0
                
                def callback(data):
                    nonlocal downloaded
                    downloaded += len(data)
                    percent = (downloaded * 100) / size
                    print(f"\rDownloading {filename}: {percent:.1f}% ({downloaded}/{size} bytes)", 
                          end='', flush=True)
                
                # Download with progress
                self.ftp.retrbinary(f'RETR {filename}', 
                                  lambda data: callback(data) or f.write(data), 
                                  blocksize=8192)
                print()  # New line after progress
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False

    def upload_to_s3(self, local_path, s3_key):
        """Upload file to S3."""
        try:
            self.s3.upload_file(local_path, self.bucket_name, s3_key)
            self.logger.info(f"Successfully uploaded to S3: {s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading to S3: {e}")
            return False

    def transfer_files(self, ftp_directory, s3_prefix):
        """Transfer all files from FTP directory to S3."""
        try:
            # Change to the specified directory
            self.logger.info(f"Changing to directory: {ftp_directory}")
            self.ftp.cwd(ftp_directory)
            
            # Get file listing
            files = []
            self.ftp.retrlines('LIST', lambda x: files.append(x.split()[-1]))
            
            # Get existing files in S3
            self.list_s3_files(s3_prefix)
            
            successful = 0
            failed = 0
            
            # Process each file
            for i, filename in enumerate(files, 1):
                # Skip if file already exists in S3
                if self.file_exists_in_s3(filename, s3_prefix):
                    self.logger.info(f"Skipping {filename} - already transferred")
                    successful += 1
                    continue
                
                print(f"Processing file {i}/{len(files)}: {filename}")
                
                # Download and upload
                local_path = os.path.join(self.temp_dir, filename)
                s3_key = os.path.join(s3_prefix, filename)
                
                if self.download_file(filename, local_path):
                    if self.upload_to_s3(local_path, s3_key):
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                
                # Clean up local file
                if os.path.exists(local_path):
                    os.remove(local_path)
            
            return successful, failed
        
        except Exception as e:
            self.logger.error(f"Error during transfer: {e}")
            return 0, 0

    def cleanup(self):
        """Clean up resources."""
        try:
            self.ftp.quit()
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")