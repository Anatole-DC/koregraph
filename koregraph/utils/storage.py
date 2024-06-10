from koregraph.models.storage import FileStorageStore


def init_file_storage(mode: str = "local") -> FileStorageStore:
    """Initialize the workflow file storage.

    Args:
        mode (str, optional): The mode to use. Defaults to "local".

    Returns:
        FileStorageInterface: The new file storage interface
    """
    return FileStorageStore(mode)
