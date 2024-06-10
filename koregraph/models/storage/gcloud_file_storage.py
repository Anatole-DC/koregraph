from dataclasses import dataclass
from .file_storage_interface import FileStorageInterface


@dataclass
class GcloudFileStorage(FileStorageInterface):
    mode = "gcloud"
