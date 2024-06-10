from pathlib import Path
from pickle import dump as pickle_dump

from tensorflow.keras.callbacks import Callback
from google.cloud.storage import Client, transfer_manager

from koregraph.utils.controllers.pickles import load_pickle_object


class GCSCallback(Callback):
    """A custom callback to copy checkpoints from local file system directory to Google Cloud Storage directory"""

    def __init__(self, cp_path: Path, bucket_name: str):
        """init method
        Args:
            cp_path (str): gcs directory path to store checkpoints
            bucket_name (str): name of GCS bucket
        """
        super(GCSCallback, self).__init__()
        self.checkpoint_path: Path = cp_path
        self.bucket_name = bucket_name

        client = Client()
        self.bucket = client.get_bucket(bucket_name)

    def upload_file_to_gcs(self, src_path: Path, dest_path: str):
        """Uploads file to Google Cloud Storage
        Args:
            src_path (str): absolute path of source file
            dest_path (str): gcs directory path beginning with 'gs://<bucket-name>'
        Returns:
        """
        # Create a complete destination path. This is basically self.cp_path + file_name.
        dest_path = dest_path + src_path.name

        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(src_path)

    def on_epoch_end(self, epoch, logs=None):
        for checkpoint_file in self.checkpoint_path.glob("*"):
            self.upload_file_to_gcs(
                src_path=checkpoint_file,
                dest_path=f"backup/{checkpoint_file.parent.name}/",
            )


class HistorySaver(Callback):
    def __init__(self, history_path):
        super(HistorySaver, self).__init__()
        self.history_path: Path = history_path

    def on_epoch_end(self, epoch: int, logs=None):
        if self.history_path.exists():
            history = load_pickle_object(self.history_path)
            for key, value in logs.items():
                history[key].append(value)
        else:
            history = {key: [value] for key, value in logs.items()}
        with open(self.history_path, "wb") as f:
            pickle_dump(history, f)
