from gcloud import storage
# from oauth2client.service_account import ServiceAccountCredentials
import os
import argparse

from utils import list_pickles

PROJECT = os.environ.get("PROJECT")
BUCKET = os.environ.get("BUCKET")


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", help="Set the preprocessing to val") #création d'un argument
    return parser.parse_args() #lancement de argparse

args = parse_arguments()

# f = vars(args)['file'].split("/")[1]
f = vars(args)['file']



pickle_list = list_pickles()

for p in pickle_list:
    name = p.split('/')[-1]
    upload_blob(BUCKET, p, name)
