from google.cloud.storage import Client, transfer_manager
from koregraph.utils.controllers.choregraphies import load_choregraphy


def download_dataset_from_bucket():
    client = Client()
    bucket = client.bucket("koregraph")
    blobs = [blob.name for blob in bucket.list_blobs()]
    transfer_manager.download_many_to_path(bucket, blobs)


if __name__ == "__main__":
    download_dataset_from_bucket()
