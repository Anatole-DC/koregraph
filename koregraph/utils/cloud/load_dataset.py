from google.cloud.storage import Client, transfer_manager

from koregraph.config.params import GCLOUD_AUTHENTICATION, BUCKET_NAME


def download_dataset_from_bucket():
    client = Client.from_service_account_json(GCLOUD_AUTHENTICATION)
    bucket = client.bucket(BUCKET_NAME)
    blobs = [blob.name for blob in bucket.list_blobs()]
    print(f"Downloading from {BUCKET_NAME}")
    transfer_manager.download_many_to_path(bucket, blobs)


if __name__ == "__main__":
    download_dataset_from_bucket()
