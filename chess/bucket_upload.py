from gcloud import storage
# from oauth2client.service_account import ServiceAccountCredentials
import os
import argparse
import parameters

from utils import list_pickles

PROJECT = parameters.PROJECT
BUCKET = parameters.BUCKET


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


dir = os.path.join(os.path.dirname(__file__), f"../pickle")
pickle_list = list_pickles(dir)

for p in pickle_list:
    name = p.split("/")[-1]
    upload_blob(BUCKET, p, name)
