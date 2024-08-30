import boto3
from datetime import datetime, timedelta
import os
import time

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'eu-north-1')
)
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ppedetectionbucket')

def cleanup_old_files(max_age_hours=1):
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        for obj in response.get('Contents', []):
            if obj['LastModified'] < datetime.now(obj['LastModified'].tzinfo) - timedelta(hours=max_age_hours):
                s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                print(f"Deleted {obj['Key']}")
    except Exception as e:
        print(f"Error cleaning up old files: {e}")

if __name__ == "__main__":
    while True:
        print("Running cleanup...")
        cleanup_old_files()
        print("Cleanup complete. Sleeping for 1 hour...")
        time.sleep(3600)  