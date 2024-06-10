from google.cloud.storage import Client, transfer_manager

from koregraph.config.params import MODEL_OUTPUT_DIRECTORY


def download_model_history_from_bucket(model_name: str):
    client = Client.from_service_account_json(
        "secrests/le-wagon-420414-c20b739bfbba.json"
    )
    bucket = client.bucket("koregraph")
    blobs = [
        blob.name
        for blob in bucket.list_blobs()
        if blob.name.startswith(f"generated/models/{model_name}")
    ]
    transfer_manager.download_many_to_path(bucket, blobs)


if __name__ == "__main__":
    download_model_history_from_bucket("test-new-save")
