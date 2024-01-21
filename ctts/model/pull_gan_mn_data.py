import pdb
import typing as t

from google.cloud import storage

BUCKET_NAME = "gan_mn"

def write_cloud_storage_bucket_to_disk(bucket_name:str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    for blob in bucket.list_blobs():
        pdb.set_trace()

if __name__ == "__main__":
    write_cloud_storage_bucket_to_disk(BUCKET_NAME)
