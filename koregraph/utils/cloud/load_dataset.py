from google.cloud.storage import Client, transfer_manager


def download_dataset_from_bucket():
    client = Client.from_service_account_json(
        "secrets/le-wagon-420414-c20b739bfbba.json"
    )
    bucket = client.bucket("koregraph")
    blobs = [blob.name for blob in bucket.list_blobs()]
    transfer_manager.download_many_to_path(bucket, blobs)


if __name__ == "__main__":
    download_dataset_from_bucket()
