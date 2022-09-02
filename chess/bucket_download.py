from copyreg import pickle
from google.cloud import storage
from utils import from_disk
import parameters

PROJECT = parameters.PROJECT
BUCKET = parameters.BUCKET


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    # print(
    #     "Downloaded storage object {} from bucket {} to local file {}.".format(
    #         source_blob_name, bucket_name, destination_file_name
    #     )


def get_pickle_name():

    pickle_names = []
    with open("bucket_pickle_list.txt", mode= "r") as f:
        d_list = f.readlines()

    for pickle_name in d_list:
        pickle_name = pickle_name.strip('\n')
        pickle_name = pickle_name.split('/')[-1]
        pickle_names.append(pickle_name)

    return pickle_names


# pickle_names = get_pickle_name()[1:]
done_pickles = []

pickle_names = ["2022-09-01_13-11-21_databatch.pkl", "2022-09-01_13-11-21_databatch2.pkl"]

for p in pickle_names:
    download_blob(BUCKET, p, 'my_unique_pikcle.pkl')
    done_pickles.append(p)
    print(from_disk("/home/njric/code/njric/chess-winner-project/my_unique_pikcle.pkl"))

print(done_pickles)

# download_blob(BUCKET, "2022-09-01_13-11-21_databatch.pkl", "./myblobdir/myblobname.pkl")

# print(type(dl_pickle))

# print(from_disk("/home/njric/code/njric/chess-winner-project/myblobdir/myblobname.pkl"))
